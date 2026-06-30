"""Measure cell length and width with a contour-based rib mesh.

The implementation follows the practical Morphometrics/MicrobeTracker-style
measurement path used by the analysis notebooks: extract a cell contour, build
a Voronoi medial graph inside that contour, resample the centerline, intersect
normal ribs with the contour boundary, and derive length and width from those
ribs.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import pi
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from scipy.spatial import QhullError, Voronoi
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import medial_axis, skeletonize

AdjacencyList = list[list[tuple[int, float]]]


@dataclass
class MeshMeasurement:
    """Container for contour-rib mesh measurements.

    Parameters:
        length_px: Pole-to-pole centerline length in pixels.
        width_px: Mean central rib width in pixels.
        length_um: Pole-to-pole centerline length in microns.
        width_um: Mean central rib width in microns.
        volume_um3: Spherocylinder volume estimate.
        surface_area_um2: Spherocylinder surface-area estimate.
        surface_area_to_volume_um_inv: Surface-area-to-volume ratio.
        method: Measurement method or failure reason.
        centerline_xy: Centerline coordinates as x/y pixel positions.
        width_profile_px: Rib-width profile in pixels.
        mesh_px: Mesh rows as left/right rib endpoint coordinates.
    """

    length_px: float
    width_px: float
    length_um: float
    width_um: float
    volume_um3: float
    surface_area_um2: float
    surface_area_to_volume_um_inv: float
    method: str
    centerline_xy: NDArray[np.float64]
    width_profile_px: NDArray[np.float64]
    mesh_px: NDArray[np.float64]

    def as_dict(self) -> dict[str, object]:
        """Convert the measurement into dataframe-friendly values.

        Returns:
            A dictionary containing scalar measurements and mesh arrays.
        """
        return {
            "length_px": self.length_px,
            "width_px": self.width_px,
            "length_um": self.length_um,
            "width_um": self.width_um,
            "volume_um3": self.volume_um3,
            "surface_area_um2": self.surface_area_um2,
            "surface_area_to_volume_um_inv": self.surface_area_to_volume_um_inv,
            "method": self.method,
            "centerline_xy": self.centerline_xy,
            "width_profile_px": self.width_profile_px,
            "mesh_px": self.mesh_px,
        }


def spherocylinder_volume(length_um: ArrayLike, width_um: ArrayLike) -> NDArray[np.float64]:
    """Calculate spherocylinder volume from pole-to-pole length and width.

    Args:
        length_um: Cell pole-to-pole length in microns.
        width_um: Cell width, interpreted as diameter, in microns.

    Returns:
        Volume in cubic microns.
    """
    length = np.asarray(length_um, dtype=float)
    width = np.asarray(width_um, dtype=float)
    radius_um = width / 2.0
    cylinder_length_um = np.maximum(length - width, 0.0)
    return pi * radius_um**2 * cylinder_length_um + (4.0 / 3.0) * pi * radius_um**3


def spherocylinder_surface_area(
    length_um: ArrayLike,
    width_um: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate spherocylinder surface area from length and width.

    Args:
        length_um: Cell pole-to-pole length in microns.
        width_um: Cell width, interpreted as diameter, in microns.

    Returns:
        Surface area in square microns.
    """
    length = np.asarray(length_um, dtype=float)
    width = np.asarray(width_um, dtype=float)
    radius_um = width / 2.0
    cylinder_length_um = np.maximum(length - width, 0.0)
    return 2.0 * pi * radius_um * cylinder_length_um + 4.0 * pi * radius_um**2


def measure_cell_mesh(
    mask_2d: ArrayLike,
    *,
    pixel_size_um: float = 1.0,
    pole_exclusion_fraction: float = 0.15,
    centerline_step_px: float = 1.0,
    contour_simplify_px: float = 0.0,
    allow_medial_axis_fallback: bool = False,
) -> dict[str, object]:
    """Measure one binary cell mask with a contour/Voronoi rib mesh.

    Args:
        mask_2d: Two-dimensional binary or binary-like cell mask.
        pixel_size_um: Pixel size in microns. Defaults to 1.0.
        pole_exclusion_fraction: Fraction of rib widths to trim from each pole
            before averaging cell width. Defaults to 0.15.
        centerline_step_px: Spacing used when resampling the ordered centerline.
            Defaults to 1.0 pixel.
        contour_simplify_px: Optional contour simplification tolerance in
            pixels. Defaults to 0.0.
        allow_medial_axis_fallback: When true, failed contour/Voronoi meshes are
            measured with a simpler distance-transform medial-axis fallback.
            Defaults to false so failures remain explicit.

    Returns:
        A dictionary containing scalar geometry measurements and mesh arrays.

    Raises:
        ValueError: If `mask_2d` is not two-dimensional.
    """
    raw_mask = np.asarray(mask_2d).astype(bool)
    if raw_mask.ndim != 2:
        raise ValueError("`mask_2d` must be a 2D array.")

    mask = _largest_component(raw_mask)
    if mask.sum() == 0:
        return _empty_measurement("empty").as_dict()

    measurement = _contour_voronoi_measurement(
        mask,
        pixel_size_um=pixel_size_um,
        pole_exclusion_fraction=pole_exclusion_fraction,
        centerline_step_px=centerline_step_px,
        contour_simplify_px=contour_simplify_px,
    )
    if measurement is not None:
        return measurement.as_dict()

    if not allow_medial_axis_fallback:
        return _empty_measurement("contour_voronoi_failed").as_dict()

    return _medial_axis_distance_transform_measurement(
        mask,
        pixel_size_um=pixel_size_um,
        pole_exclusion_fraction=pole_exclusion_fraction,
    ).as_dict()


def _contour_voronoi_measurement(
    mask: NDArray[np.bool_],
    pixel_size_um: float,
    pole_exclusion_fraction: float,
    centerline_step_px: float,
    contour_simplify_px: float,
) -> MeshMeasurement | None:
    contour_xy = _mask_contour_xy(mask)
    if contour_xy is None or len(contour_xy) < 4:
        return None

    polygon = _polygon_from_contour(contour_xy, contour_simplify_px=contour_simplify_px)
    if polygon is None or polygon.is_empty or polygon.area <= 0:
        return None

    graph = _inside_voronoi_graph(contour_xy, polygon)
    if graph.number_of_edges() == 0:
        return None

    centerline_xy = _ordered_centerline_from_graph(graph)
    if centerline_xy.shape[0] < 2:
        return None

    centerline_xy = _resample_polyline(centerline_xy, step_px=centerline_step_px)
    if centerline_xy.shape[0] < 2:
        return None

    mesh_px = _mesh_from_centerline_and_boundary(centerline_xy, polygon, max(mask.shape) * 3.0)
    if mesh_px.shape[0] < 5:
        return None

    centerline_xy = np.column_stack(
        [
            (mesh_px[:, 0] + mesh_px[:, 2]) / 2.0,
            (mesh_px[:, 1] + mesh_px[:, 3]) / 2.0,
        ]
    )
    width_profile_px = np.sqrt(
        (mesh_px[:, 0] - mesh_px[:, 2]) ** 2
        + (mesh_px[:, 1] - mesh_px[:, 3]) ** 2
    )
    step_lengths_px = np.sqrt(np.sum(np.diff(centerline_xy, axis=0) ** 2, axis=1))
    length_px = float(step_lengths_px.sum())
    width_px = float(_central_width(width_profile_px, pole_exclusion_fraction))

    if not np.isfinite(length_px) or not np.isfinite(width_px) or length_px <= 0 or width_px <= 0:
        return None

    return _measurement_from_length_width(
        length_px=length_px,
        width_px=width_px,
        pixel_size_um=pixel_size_um,
        method="contour_voronoi_rib_intersections",
        centerline_xy=centerline_xy,
        width_profile_px=width_profile_px,
        mesh_px=mesh_px,
    )


def _medial_axis_distance_transform_measurement(
    mask: NDArray[np.bool_],
    pixel_size_um: float,
    pole_exclusion_fraction: float,
) -> MeshMeasurement:
    skeleton, distance = _medial_skeleton(mask)
    centerline_rc = _longest_skeleton_path(skeleton)
    if centerline_rc.shape[0] < 2:
        return _regionprops_fallback(mask, pixel_size_um)

    step_lengths_px = np.sqrt(np.sum(np.diff(centerline_rc.astype(float), axis=0) ** 2, axis=1))
    length_px = float(step_lengths_px.sum())
    width_profile_px = 2.0 * distance[centerline_rc[:, 0], centerline_rc[:, 1]]
    width_px = float(_central_width(width_profile_px, pole_exclusion_fraction))

    centerline_xy = centerline_rc[:, [1, 0]].astype(float)
    mesh_px = _mesh_from_centerline(centerline_xy, width_profile_px)
    return _measurement_from_length_width(
        length_px=length_px,
        width_px=width_px,
        pixel_size_um=pixel_size_um,
        method="medial_axis_distance_transform",
        centerline_xy=centerline_xy,
        width_profile_px=width_profile_px,
        mesh_px=mesh_px,
    )


def _mask_contour_xy(mask: NDArray[np.bool_]) -> NDArray[np.float64] | None:
    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0)
    contours = find_contours(padded, level=0.5)
    if not contours:
        return None

    contour_rc = max(contours, key=len) - 1.0
    contour_xy = contour_rc[:, [1, 0]].astype(float)
    if len(contour_xy) > 1 and np.allclose(contour_xy[0], contour_xy[-1]):
        contour_xy = contour_xy[:-1]
    contour_xy = _drop_duplicate_points(contour_xy)
    if len(contour_xy) < 4:
        return None
    return contour_xy


def _drop_duplicate_points(
    points: NDArray[np.float64],
    decimals: int = 8,
) -> NDArray[np.float64]:
    if len(points) == 0:
        return points
    rounded = np.round(points, decimals=decimals)
    _, keep_indices = np.unique(rounded, axis=0, return_index=True)
    keep_indices = np.sort(keep_indices)
    points = points[keep_indices]
    if len(points) > 1:
        step = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        keep = np.r_[True, step > 1e-9]
        points = points[keep]
    return points


def _polygon_from_contour(
    contour_xy: NDArray[np.float64],
    contour_simplify_px: float = 0.0,
) -> Polygon | None:
    polygon = Polygon(contour_xy)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    if contour_simplify_px > 0:
        polygon = polygon.simplify(contour_simplify_px, preserve_topology=True)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if polygon.geom_type != "Polygon":
        return None
    return polygon


def _inside_voronoi_graph(contour_xy: NDArray[np.float64], polygon: Polygon) -> nx.Graph:
    graph = nx.Graph()
    try:
        vor = Voronoi(contour_xy)
    except QhullError:
        return graph

    buffered_polygon = polygon.buffer(1e-7)
    for ridge in vor.ridge_vertices:
        if len(ridge) != 2 or ridge[0] < 0 or ridge[1] < 0:
            continue
        p0 = vor.vertices[ridge[0]]
        p1 = vor.vertices[ridge[1]]
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            continue
        if np.linalg.norm(p1 - p0) < 1e-9:
            continue

        line = LineString([tuple(p0), tuple(p1)])
        if line.is_empty or line.length < 1e-9:
            continue
        if not buffered_polygon.covers(Point(float(p0[0]), float(p0[1]))):
            continue
        if not buffered_polygon.covers(Point(float(p1[0]), float(p1[1]))):
            continue
        if not buffered_polygon.covers(line):
            continue

        n0 = _node_key(p0)
        n1 = _node_key(p1)
        graph.add_node(n0, xy=np.asarray(p0, dtype=float))
        graph.add_node(n1, xy=np.asarray(p1, dtype=float))
        graph.add_edge(n0, n1, weight=float(line.length))

    return _largest_weighted_component(_prune_terminal_branches(graph))


def _node_key(point: NDArray[np.float64], decimals: int = 8) -> tuple[float, float]:
    rounded = np.round(np.asarray(point, dtype=float), decimals=decimals)
    return float(rounded[0]), float(rounded[1])


def _copy_edge_subgraph(graph: nx.Graph, edges: list[tuple[Any, Any]]) -> nx.Graph:
    out = nx.Graph()
    for u, v in edges:
        out.add_node(u, xy=graph.nodes[u]["xy"])
        out.add_node(v, xy=graph.nodes[v]["xy"])
        out.add_edge(u, v, weight=graph.edges[u, v]["weight"])
    return out


def _prune_terminal_branches(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_edges() == 0:
        return graph

    current = graph.copy()
    while current.number_of_edges() > 0:
        degree = dict(current.degree())
        keep_edges = [
            (u, v)
            for u, v in current.edges()
            if degree.get(u, 0) > 1 and degree.get(v, 0) > 1
        ]
        removed = current.number_of_edges() - len(keep_edges)
        if removed <= 2:
            pruned = _copy_edge_subgraph(current, keep_edges)
            return pruned if pruned.number_of_edges() else current
        current = _copy_edge_subgraph(current, keep_edges)

    return graph


def _largest_weighted_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_edges() == 0:
        return graph
    components = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph)]
    return max(
        components,
        key=lambda comp: sum(data.get("weight", 1.0) for _, _, data in comp.edges(data=True)),
    )


def _ordered_centerline_from_graph(graph: nx.Graph) -> NDArray[np.float64]:
    if graph.number_of_nodes() < 2:
        return np.empty((0, 2), dtype=float)

    endpoints = [node for node, degree in graph.degree() if degree == 1]
    candidate_nodes = endpoints if len(endpoints) >= 2 else list(graph.nodes)

    best_distance = -np.inf
    best_path: list[Any] | None = None
    for start in candidate_nodes:
        lengths, paths = nx.single_source_dijkstra(graph, start, weight="weight")
        for end in candidate_nodes:
            if end == start or end not in lengths:
                continue
            distance = float(lengths[end])
            if distance > best_distance:
                best_distance = distance
                best_path = paths[end]

    if best_path is None:
        return np.empty((0, 2), dtype=float)
    return np.asarray([graph.nodes[node]["xy"] for node in best_path], dtype=float)


def _resample_polyline(
    points: NDArray[np.float64],
    step_px: float = 1.0,
) -> NDArray[np.float64]:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points

    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    length = float(distances.sum())
    if not np.isfinite(length) or length <= 0:
        return points[:1]

    cumulative = np.r_[0.0, np.cumsum(distances)]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    cumulative = cumulative[keep]
    points = points[keep]
    if len(points) < 2:
        return points

    sample = np.arange(0.0, length, step_px)
    if len(sample) == 0 or sample[-1] < length:
        sample = np.r_[sample, length]
    x = np.interp(sample, cumulative, points[:, 0])
    y = np.interp(sample, cumulative, points[:, 1])
    return np.column_stack([x, y])


def _mesh_from_centerline_and_boundary(
    centerline_xy: NDArray[np.float64],
    polygon: Polygon,
    rib_length_px: float,
) -> NDArray[np.float64]:
    centerline_xy = np.asarray(centerline_xy, dtype=float)
    if centerline_xy.shape[0] < 2:
        return np.empty((0, 4), dtype=float)

    start_tangent = _unit_vector(centerline_xy[1] - centerline_xy[0])
    end_tangent = _unit_vector(centerline_xy[-1] - centerline_xy[-2])
    if start_tangent is None or end_tangent is None:
        return np.empty((0, 4), dtype=float)

    start_pole = _pole_intersection(centerline_xy[0], start_tangent, polygon, rib_length_px, sign=-1)
    end_pole = _pole_intersection(centerline_xy[-1], end_tangent, polygon, rib_length_px, sign=1)
    if start_pole is None or end_pole is None:
        return np.empty((0, 4), dtype=float)

    rows = [[start_pole[0], start_pole[1], start_pole[0], start_pole[1]]]
    for point, tangent in zip(centerline_xy[1:-1], _centerline_tangents(centerline_xy)[1:-1]):
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        rib = _rib_intersections(point, normal, polygon, rib_length_px)
        if rib is not None:
            rows.append([rib[0][0], rib[0][1], rib[1][0], rib[1][1]])
    rows.append([end_pole[0], end_pole[1], end_pole[0], end_pole[1]])
    return np.asarray(rows, dtype=float)


def _centerline_tangents(centerline_xy: NDArray[np.float64]) -> NDArray[np.float64]:
    tangent = np.empty_like(centerline_xy, dtype=float)
    tangent[1:-1] = centerline_xy[2:] - centerline_xy[:-2]
    tangent[0] = centerline_xy[1] - centerline_xy[0]
    tangent[-1] = centerline_xy[-1] - centerline_xy[-2]
    norms = np.linalg.norm(tangent, axis=1)
    norms[norms == 0] = 1.0
    return tangent / norms[:, None]


def _unit_vector(vector: NDArray[np.float64]) -> NDArray[np.float64] | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return None
    return np.asarray(vector, dtype=float) / norm


def _pole_intersection(
    point: NDArray[np.float64],
    tangent: NDArray[np.float64],
    polygon: Polygon,
    rib_length_px: float,
    sign: int,
) -> NDArray[np.float64] | None:
    line = LineString(
        [
            tuple(point - tangent * rib_length_px),
            tuple(point + tangent * rib_length_px),
        ]
    )
    candidates = _points_from_geometry(line.intersection(polygon.boundary))
    if not candidates:
        return None
    projections = np.asarray([np.dot(candidate - point, tangent) for candidate in candidates])
    if sign < 0:
        valid = np.flatnonzero(projections <= 1e-7)
        index = valid[np.argmin(projections[valid])] if len(valid) else int(np.argmin(projections))
    else:
        valid = np.flatnonzero(projections >= -1e-7)
        index = valid[np.argmax(projections[valid])] if len(valid) else int(np.argmax(projections))
    return candidates[int(index)]


def _rib_intersections(
    point: NDArray[np.float64],
    normal: NDArray[np.float64],
    polygon: Polygon,
    rib_length_px: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    line = LineString(
        [
            tuple(point - normal * rib_length_px),
            tuple(point + normal * rib_length_px),
        ]
    )
    candidates = _points_from_geometry(line.intersection(polygon.boundary))
    if len(candidates) < 2:
        return None

    projections = np.asarray([np.dot(candidate - point, normal) for candidate in candidates])
    negative = np.flatnonzero(projections < -1e-7)
    positive = np.flatnonzero(projections > 1e-7)
    if len(negative) and len(positive):
        neg = negative[np.argmax(projections[negative])]
        pos = positive[np.argmin(projections[positive])]
        return candidates[int(pos)], candidates[int(neg)]

    order = np.argsort(projections)
    if projections[order[-1]] - projections[order[0]] <= 1e-7:
        return None
    return candidates[int(order[-1])], candidates[int(order[0])]


def _points_from_geometry(geometry: BaseGeometry) -> list[NDArray[np.float64]]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [np.asarray(geometry.coords[0], dtype=float)]
    if isinstance(geometry, MultiPoint):
        return [np.asarray(point.coords[0], dtype=float) for point in geometry.geoms]
    if isinstance(geometry, LineString):
        coords = np.asarray(geometry.coords, dtype=float)
        return [coords[0], coords[-1]]
    if isinstance(geometry, MultiLineString):
        points: list[NDArray[np.float64]] = []
        for line in geometry.geoms:
            points.extend(_points_from_geometry(line))
        return points
    if isinstance(geometry, GeometryCollection):
        points = []
        for geom in geometry.geoms:
            points.extend(_points_from_geometry(geom))
        return points
    return []


def _largest_component(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    labeled = label(mask, connectivity=2)
    props = regionprops(labeled)
    if not props:
        return np.zeros_like(mask, dtype=bool)
    largest = max(props, key=lambda prop: prop.area)
    return labeled == largest.label


def _medial_skeleton(mask: NDArray[np.bool_]) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    skeleton, distance = medial_axis(mask, return_distance=True, rng=0)
    if skeleton.sum() < 2:
        skeleton = skeletonize(mask)
        distance = ndi.distance_transform_edt(mask)
    return skeleton.astype(bool), np.asarray(distance, dtype=float)


def _longest_skeleton_path(skeleton: NDArray[np.bool_]) -> NDArray[np.int64]:
    coords = np.argwhere(skeleton)
    if coords.shape[0] < 2:
        return coords

    index = {tuple(coord): i for i, coord in enumerate(coords)}
    adjacency: AdjacencyList = [[] for _ in range(coords.shape[0])]
    for i, (row, col) in enumerate(coords):
        row_int = int(row)
        col_int = int(col)
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                j = index.get((row_int + drow, col_int + dcol))
                if j is not None:
                    adjacency[i].append((j, float(np.hypot(drow, dcol))))

    degrees = np.array([len(neighbors) for neighbors in adjacency])
    endpoints = np.flatnonzero(degrees == 1)
    if endpoints.size >= 2:
        start, end, parent = _longest_endpoint_pair(adjacency, endpoints)
    else:
        start, end, parent = _longest_two_pass_path(adjacency)

    path_indices = _reconstruct_path(parent, start, end)
    if len(path_indices) < 2:
        return coords
    return coords[np.asarray(path_indices, dtype=int)]


def _dijkstra(adjacency: AdjacencyList, start: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    distances = np.full(len(adjacency), np.inf, dtype=float)
    parents = np.full(len(adjacency), -1, dtype=int)
    distances[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        dist, node = heappop(heap)
        if dist > distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                parents[neighbor] = node
                heappush(heap, (new_dist, neighbor))
    return distances, parents


def _longest_endpoint_pair(
    adjacency: AdjacencyList,
    endpoints: NDArray[np.int64],
) -> tuple[int, int, NDArray[np.int64]]:
    best_start = int(endpoints[0])
    best_end = int(endpoints[1])
    best_parent = np.full(len(adjacency), -1, dtype=int)
    best_distance = -np.inf
    endpoint_set = set(int(endpoint) for endpoint in endpoints)
    for start in endpoint_set:
        distances, parents = _dijkstra(adjacency, start)
        for end in endpoint_set:
            if end != start and np.isfinite(distances[end]) and distances[end] > best_distance:
                best_start = start
                best_end = end
                best_parent = parents
                best_distance = distances[end]
    return best_start, best_end, best_parent


def _longest_two_pass_path(adjacency: AdjacencyList) -> tuple[int, int, NDArray[np.int64]]:
    distances, _ = _dijkstra(adjacency, 0)
    finite = np.flatnonzero(np.isfinite(distances))
    start = int(finite[np.argmax(distances[finite])])
    distances, parents = _dijkstra(adjacency, start)
    finite = np.flatnonzero(np.isfinite(distances))
    end = int(finite[np.argmax(distances[finite])])
    return start, end, parents


def _reconstruct_path(parent: NDArray[np.int64], start: int, end: int) -> list[int]:
    path = [end]
    node = end
    while node != start and node >= 0:
        node = int(parent[node])
        if node >= 0:
            path.append(node)
    if path[-1] != start:
        return []
    path.reverse()
    return path


def _central_width(width_profile_px: NDArray[np.float64], pole_exclusion_fraction: float) -> float:
    width_profile = np.asarray(width_profile_px, dtype=float)
    valid = np.isfinite(width_profile) & (width_profile > 0)
    if not valid.any():
        return np.nan
    profile = width_profile[valid]
    n_points = len(profile)
    trim = int(np.floor(n_points * pole_exclusion_fraction))
    if trim > 0 and n_points > 2 * trim + 1:
        profile = profile[trim:-trim]
    return float(np.mean(profile))


def _mesh_from_centerline(
    centerline_xy: NDArray[np.float64],
    width_profile_px: NDArray[np.float64],
) -> NDArray[np.float64]:
    if centerline_xy.shape[0] < 2:
        return np.empty((0, 4), dtype=float)

    tangent = np.empty_like(centerline_xy, dtype=float)
    tangent[1:-1] = centerline_xy[2:] - centerline_xy[:-2]
    tangent[0] = centerline_xy[1] - centerline_xy[0]
    tangent[-1] = centerline_xy[-1] - centerline_xy[-2]

    norm = np.linalg.norm(tangent, axis=1)
    norm[norm == 0] = 1.0
    tangent = tangent / norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    half_width = np.asarray(width_profile_px, dtype=float)[:, None] / 2.0
    side_a = centerline_xy + normal * half_width
    side_b = centerline_xy - normal * half_width
    return np.column_stack([side_a[:, 0], side_a[:, 1], side_b[:, 0], side_b[:, 1]])


def _measurement_from_length_width(
    length_px: float,
    width_px: float,
    pixel_size_um: float,
    method: str,
    centerline_xy: NDArray[np.float64],
    width_profile_px: NDArray[np.float64],
    mesh_px: NDArray[np.float64],
) -> MeshMeasurement:
    length_um = float(length_px * pixel_size_um)
    width_um = float(width_px * pixel_size_um)
    volume_um3 = float(np.asarray(spherocylinder_volume(length_um, width_um)))
    surface_area_um2 = float(np.asarray(spherocylinder_surface_area(length_um, width_um)))
    if volume_um3 > 0:
        surface_area_to_volume_um_inv = surface_area_um2 / volume_um3
    else:
        surface_area_to_volume_um_inv = np.nan
    return MeshMeasurement(
        length_px=float(length_px),
        width_px=float(width_px),
        length_um=length_um,
        width_um=width_um,
        volume_um3=volume_um3,
        surface_area_um2=surface_area_um2,
        surface_area_to_volume_um_inv=float(surface_area_to_volume_um_inv),
        method=method,
        centerline_xy=np.asarray(centerline_xy, dtype=float),
        width_profile_px=np.asarray(width_profile_px, dtype=float),
        mesh_px=np.asarray(mesh_px, dtype=float),
    )


def _regionprops_fallback(mask: NDArray[np.bool_], pixel_size_um: float) -> MeshMeasurement:
    props = regionprops(label(mask, connectivity=2))
    if not props:
        return _empty_measurement("empty")
    region = max(props, key=lambda prop: prop.area)
    return _measurement_from_length_width(
        length_px=float(region.axis_major_length),
        width_px=float(region.axis_minor_length),
        pixel_size_um=pixel_size_um,
        method="regionprops_fallback",
        centerline_xy=np.empty((0, 2), dtype=float),
        width_profile_px=np.empty(0, dtype=float),
        mesh_px=np.empty((0, 4), dtype=float),
    )


def _empty_measurement(method: str) -> MeshMeasurement:
    return MeshMeasurement(
        length_px=np.nan,
        width_px=np.nan,
        length_um=np.nan,
        width_um=np.nan,
        volume_um3=np.nan,
        surface_area_um2=np.nan,
        surface_area_to_volume_um_inv=np.nan,
        method=method,
        centerline_xy=np.empty((0, 2), dtype=float),
        width_profile_px=np.empty(0, dtype=float),
        mesh_px=np.empty((0, 4), dtype=float),
    )


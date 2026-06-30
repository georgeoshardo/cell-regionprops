# cell-regionprops

Cell-focused `regionprops_table`-style measurements for labeled masks.

The core API measures one labeled 2D mask and returns one dataframe row per
label. Binary masks are handled by converting foreground pixels to label 1.
Batch helpers apply the same single-frame function over stacks whose last two
axes are `y, x`.


"""
Grid class
when the top/bottom rows and left/right
columns are treated as being adjacent
"""
row, col = 1, 0
_grid_height, _grid_width = 6, 9


up = (row - 1) % (_grid_height )
down = (row + 1) % (_grid_height )
left = (col - 1) % (_grid_width )
right = (col + 1) % (_grid_width )


print [[up, col], [down, col], [row, left], [row, right]]
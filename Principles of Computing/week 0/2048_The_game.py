"""
Clone of 2048 game.
"""

import poc_2048_gui
import random

# Directions, DO NOT MODIFY
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Offsets for computing tile indices in each direction.
# DO NOT MODIFY this dictionary.    
OFFSETS = {UP: (1, 0), 
           DOWN: (-1, 0), 
           LEFT: (0, 1), 
           RIGHT: (0, -1)} 
   
def shift(line):
    """
    Helper function that shifts non-zero elements of line[] to the left
    """
    idx, shifted_list = 0, [0 for idx in range(len(line))]
    for i_idx in range(len(line)):
        if line[i_idx] != 0:
            shifted_list[idx] = line[i_idx]
            idx += 1
    return shifted_list   
     
def merge(line):
    """
    Helper function that merges a single row or column in 2048
    """      
    result = shift(line)
    
    for i_idx in range(len(result) - 1):
        if (result[i_idx] != 0) and (result[i_idx] == result[i_idx + 1]):
            result[i_idx] = result[i_idx] + result[i_idx + 1]
            result[i_idx + 1] = 0
      
    return shift(result)                

class TwentyFortyEight:
    """
    Class to run the game logic.
    """

    def __init__(self, grid_height, grid_width):
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        lst_up 		= ([[0, i_idx] for i_idx in range(self.grid_width)])
        lst_down 	= ([[self.grid_height - 1, i_idx] for i_idx in range(self.grid_width)])
        lst_left 	= ([[i_idx, 0] for i_idx in range(self.grid_height)])
        lst_right 	= ([[i_idx, self.grid_width - 1] for i_idx in range(self.grid_height)])
  
        self.directions = {UP: lst_up,
                           DOWN : lst_down,
                           LEFT : lst_left,
                           RIGHT : lst_right}
        self.reset()

    def reset(self):
        """
        Reset the game so the grid is empty.
        """       
        self.grid_2d = [[0 for dummy_col in range(self.grid_width)] for dummy_row in range(self.grid_height)]
           
    def __str__(self):
        """
        Return a string representation of the grid for debugging.
        """
        grid = ""
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                grid = grid + "[" + str(row) + ", " + str(col) + "] = " + str(self.grid_2d[row][col]) + " " 
            grid += "\n"
        return grid

    def get_grid_height(self):
        """
        Get the height of the board.
        """
        return self.grid_height
    
    def get_grid_width(self):
        """
        Get the width of the board.
        """
        return self.grid_width
                            
    def move(self, direction):
        """
        Move all tiles in the given direction and add
        a new tile if any tiles moved.
        """
        merged_temp = []
        
        temp_list = [[None for dummy_col in range(self.grid_width)] for dummy_row in range(self.grid_height)] 
        for i_idx in range(self.grid_height):
            for j_idx in range(self.grid_width):
                temp_list[i_idx][j_idx] = self.grid_2d[i_idx][j_idx]
        
        if (direction == UP):
            for idx in self.directions[UP]:
                temp_col = []
                for row in range(self.grid_height):
                    temp_col.append(self.grid_2d[idx[0] + row] [idx[1]])
                merged_temp.append(merge(temp_col))
                
        elif (direction == DOWN):
            for idx in self.directions[DOWN]:
                temp_col = []
                for row in range(self.grid_height):
                    temp_col.append(self.grid_2d[idx[0] - row] [idx[1]])
                merged_temp.append(merge(temp_col))
                
        elif (direction == LEFT):
            for idx in self.directions[LEFT]:
                merged_temp.append(merge(self.grid_2d[idx[0]]))
        
        elif (direction == RIGHT):
            for idx in self.directions[RIGHT]:
                temp_col = (self.grid_2d[idx[0]])
                temp_col.reverse()
                temp_col = merge(temp_col)
                merged_temp.append(temp_col)

        for i_idx in range(self.grid_height):
            for j_idx in range(self.grid_width):
                if (direction == UP):
                    self.grid_2d[i_idx][j_idx] = merged_temp[j_idx][i_idx]
                elif (direction == DOWN):
                    self.grid_2d[i_idx][j_idx] = merged_temp[j_idx][self.grid_height -1 - i_idx]
                elif (direction == LEFT):
                    self.grid_2d[i_idx][j_idx] = merged_temp[i_idx][j_idx]
                elif (direction == RIGHT):
                    self.grid_2d[i_idx][j_idx] = merged_temp[i_idx][self.grid_width -1 - j_idx]
                                
        if temp_list != self.grid_2d:
            TwentyFortyEight.new_tile(self)             
            
        return self.grid_2d          

    def new_tile(self):
        """
        Create a new tile in a randomly selected empty 
        square.  The tile should be 2 90% of the time and
        4 10% of the time.
        """
        has_available_tiles = False
        empty_tiles = []         
        
        for row in range(0, self.grid_height):
            for col in range(0, self.grid_width):
                if self.grid_2d[row][col] == 0:
                    empty_tiles.append([row,col])
                    has_available_tiles = True
                    
        if has_available_tiles:
            random_idx = random.choice(empty_tiles)
            if random.randint(0, 9) < 9:
                self.set_tile(random_idx[0],random_idx[1],2)
            else:
                self.set_tile(random_idx[0],random_idx[1],4)
        
    def set_tile(self, row, col, value):
        """
        Set the tile at position row, col to have the given value.
        """        
        self.grid_2d[row][col] = value
        
        return self.grid_2d[row][col]

    def get_tile(self, row, col):
        """
        Return the value of the tile at position row, col.
        """        
        return self.grid_2d[row][col]
 
    
poc_2048_gui.run_gui(TwentyFortyEight(4, 4))
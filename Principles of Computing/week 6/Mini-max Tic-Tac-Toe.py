"""
Mini-max Tic-Tac-Toe Player
"""

import poc_ttt_gui
import poc_ttt_provided as provided

# Set timeout, as mini-max can take a long time
import codeskulptor
codeskulptor.set_timeout(60)

# SCORING VALUES - DO NOT MODIFY
SCORES = {provided.PLAYERX: 1,
          provided.DRAW: 0,
          provided.PLAYERO: -1}

def mm_move(board, player):
    """
    Make a move on the board.
    
    Returns a tuple with two elements.  The first element is the score
    of the given board and the second element is the desired move as a
    tuple, (row, col).
    """
    check_res = board.check_win()  
    if check_res != None:  
        return SCORES[check_res], (-1,-1)  
    else:  
        empty_list = board.get_empty_squares()  
        com_score = -2  
        max_score = -2  
        max_each = (-1,-1)  
        changed_player = provided.switch_player(player)  
        for square in empty_list:  
            cur_board = board.clone()  
            cur_board.move(square[0], square[1], player)  
            cur_score_tuple = mm_move(cur_board, changed_player)  
            cur_score = cur_score_tuple[0]  
            if cur_score * SCORES[player] > com_score:  
                com_score = cur_score * SCORES[player] # used for compare  
                max_score = cur_score  # used for return a value  
                max_each = square  
            if com_score == 1:  
                return max_score, max_each              
    return max_score, max_each  

def move_wrapper(board, player, trials):
    """
    Wrapper to allow the use of the same infrastructure that was used
    for Monte Carlo Tic-Tac-Toe.
    """
    move = mm_move(board, player)
    assert move[1] != (-1, -1), "returned illegal move (-1, -1)"
    return move[1]

# Test game with the console or the GUI.
# Uncomment whichever you prefer.
# Both should be commented out when you submit for
# testing to save time.

# provided.play_game(move_wrapper, 1, False)        
poc_ttt_gui.run_gui(3, provided.PLAYERO, move_wrapper, 1, False)

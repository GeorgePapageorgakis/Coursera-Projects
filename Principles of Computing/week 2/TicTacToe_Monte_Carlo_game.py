"""
Monte Carlo Tic-Tac-Toe Player
"""

import random
import poc_ttt_gui
import poc_ttt_provided as provided

# Constants for Monte Carlo simulator
# Change as desired
NTRIALS = 80    # Number of trials to run
MCMATCH = 1.0  # Score for squares played by the machine player
MCOTHER = 1.0  # Score for squares played by the other player
    
def mc_trial(board, player):
    """
    The function plays a game starting with the given player by 
    making random moves. Returns when the game is over. Modified board
    contains the state of the game, so it does not return anything.
    """
    while board.check_win() == None:
        available_squares = board.get_empty_squares()
        random_square = random.choice(available_squares)
        board.move(random_square[0], random_square[1], player)
        provided.switch_player(player)

def mc_update_scores(scores, board, player):
    """
    Scores the completed board and update the scores grid.
    """
    if board.check_win() != provided.DRAW:
        for idx in range(board.get_dim()):
            for jdx in range(board.get_dim()):
                if board.square(idx,jdx) == player:
                    if board.check_win() == player:
                        scores[idx][jdx] += MCMATCH
                    else:
                        scores[idx][jdx] -= MCMATCH
                elif board.square(idx,jdx) == provided.switch_player(player):
                    if board.check_win() == provided.switch_player(player):
                        scores[idx][jdx] += MCOTHER
                    else:
                        scores[idx][jdx] -= MCOTHER

def get_best_move(board, scores):
    """
    Find all of the empty squares with the maximum score and 
    randomly return one of them as a (row, column) tuple. It 
    is an error to call this function with a board that is full
    """
    max_score = float('-inf')
    max_score_pos = (0, 0)
    available_squares = board.get_empty_squares()
    for idx in available_squares:
        if scores[idx[0]][idx[1]] > max_score:
            max_score = scores[idx[0]][idx[1]]
            max_score_pos = idx
    
    return max_score_pos

def mc_move(board, player, trials):
    """
    Use the Monte Carlo simulation to return a move for the machine
    player in the form of a (row, column) tuple.
    """
    scores = [ [0 for dummy_col in range(board.get_dim())] for dummy_row in range(board.get_dim())]
	
	# For obvious player winning move
    for dummy_idx in board.get_empty_squares():
        temp_obj = board.clone()
        temp_obj.move(dummy_idx[0], dummy_idx[1],  player)
        if temp_obj.check_win() == player:
            return dummy_idx
    
    # For obvious opponent winning move
    for dummy_idx in board.get_empty_squares():
        temp_obj = board.clone()
        temp_obj.move(dummy_idx[0], dummy_idx[1],  provided.switch_player(player))
        if temp_obj.check_win() == provided.switch_player(player):
            return dummy_idx
			
    # Random move
    for dummy_idx in range(trials):
        temp_obj = board.clone()
        mc_trial(temp_obj, player)
        mc_update_scores(scores, temp_obj, player)
    
    if board.get_empty_squares() != []:
        return get_best_move(board, scores)
    

# Test game with the console or the GUI.
# Uncomment whichever you prefer.
# Both should be commented out when you submit for
# testing to save time.

provided.play_game(mc_move, NTRIALS, False)        
poc_ttt_gui.run_gui(3, provided.PLAYERX, mc_move, NTRIALS, False)

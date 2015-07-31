# template for "Guess the number" mini-project
# input will come from buttons and an input field
# all output for the game will be printed in the console

import random 
import simplegui
import math

# initialize global variables used in your code  
canvas_width = 200  
canvas_height = 200  
control_width = 120  
object_width = 120 

guess_range = 100
remain_guess = 0
rand_num = 0


# helper function to start and restart the game
def new_game():
    global rand_num, remain_guess
    #generate a random number
    rand_num = random.randrange(0, guess_range)
    #calculate number of guesses
    remain_guess = int(math.ceil(math.log(guess_range + 1, 2)))
    print 'New game, number' + "'" + 's range is [ 0,',guess_range,').'  
    print 'You have ', remain_guess, ' chances to guess.\n' 

# define event handlers for control panel
def range100():
    # button that changes range to range [0,100) and restarts
    global guess_range 
    guess_range = 100  
    new_game()

def range1000():
    # button that changes range to range [0,1000) and restarts
    global guess_range 
    guess_range = 1000  
    new_game()
    
def input_guess(guess):
    # main game logic goes here
    
    global remain_guess
    guess_num = int(guess)  
      
    print 'Your guess was: ', guess_num,'\n',  
     
    if (guess_num >= 0) and (guess_num < guess_range):
        remain_guess -= 1
        print 'Number of remaining guesses is ', remain_guess,''
        if guess_num == rand_num:  
            print 'Correct!\n'  
            new_game()  
        elif guess_num < rand_num:  
            print 'Higher!\n'  
        else:  
            print 'Lower!\n'        
        if remain_guess <= 0:  
            print 'You used up guess chances! Game over!\n'  
            new_game()
    else:
        print 'Invalid guess number range. Try again!'  
        print 'Number of remaining guesses is ', remain_guess,'\n'
    
# create frame
frame = simplegui.create_frame('Guess the number', canvas_width, canvas_height,control_width)


# register event handlers for control elements  
frame.add_button('Range: [0, 100)', range100, object_width)  
frame.add_button('Range: [0, 1000)', range1000, object_width)
frame.add_input('Enter your guess', input_guess, object_width)


# call new_game and start frame
new_game() 
frame.start()


# always remember to check your completed program against the grading rubric

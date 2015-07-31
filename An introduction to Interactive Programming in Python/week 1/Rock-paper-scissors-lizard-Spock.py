# Rock-paper-scissors-lizard-Spock template


# The key idea of this program is to equate the strings
# "rock", "paper", "scissors", "lizard", "Spock" to numbers
# as follows:
#
# 0 - rock
# 1 - Spock
# 2 - paper
# 3 - lizard
# 4 - scissors

# helper functions
from random import randrange

def name_to_number(name):
    # convert name to a number using if/elif/else
    if name == "rock":
        return 0
    elif name == "Spock":
        return 1
    elif name == "paper":
        return 2
    elif name == "lizard":
        return 3
    elif name == "scissors":
        return 4
    else:
        return 5

def number_to_name(number):
    # convert number to name using if/elif/else
    if number == 0:
        return "rock"
    elif number == 1:
        return "Spock"
    elif number == 2:
        return "paper"
    elif number == 3:
        return "lizard"
    elif number == 4:
        return "scissors"
    else: 
        print "Invalid input number: " + number + "\n"

def rpsls(player_choice):
    # convert the player's choice to player_number using the function name_to_number()
    player_number = name_to_number(player_choice)
    
    #if name entered is invalid.
    if player_number == 5:
        print "Invalid input option: " + player_choice + "\n"
        return
    
    # compute random guess for comp_number using random.randrange()
    comp_number = randrange(0, 5)
    
    # print out the message for the player's choice
    print 'Player chooses', player_choice
    
    # convert comp_number to comp_choice using the function number_to_name()
    # print out the message for computer's choice
    print 'Computer chooses', number_to_name(comp_number)

    # compute difference of comp_number and player_number modulo five
    # use if/elif/else to determine winner, print winner message
    if (player_number - comp_number) % 5 >= 3: 
        print 'Computer wins!\n'
    elif (player_number - comp_number) % 5 == 0: 
        print 'Player and computer tie!\n'
    else:
        print 'Player wins!\n'
    
# test your code - LEAVE THESE CALLS IN YOUR SUBMITTED CODE
rpsls("rock")
rpsls("Spock")
rpsls("paper")
rpsls("lizard")
rpsls("scissors")

# always remember to check your completed program against the grading rubric



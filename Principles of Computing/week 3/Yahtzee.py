"""
Planner for Yahtzee
Simplifications:  only allow discard and roll, only score against upper level
"""

# Used to increase the timeout, if necessary
import codeskulptor
import random
codeskulptor.set_timeout(20)

def gen_all_sequences(outcomes, length):
    """
    Iterative function that enumerates the set of all sequences of
    outcomes of given length.
    """
    
    answer_set = set([()])
    for dummy_idx in range(length):
        temp_set = set()
        for partial_sequence in answer_set:
            for item in outcomes:
                new_sequence = list(partial_sequence)
                new_sequence.append(item)
                temp_set.add(tuple(new_sequence))
        answer_set = temp_set
    return answer_set


def score(hand):
    """
    Compute the maximal score for a Yahtzee hand according to the
    upper section of the Yahtzee score card.

    hand: full yahtzee hand

    Returns an integer score 
    """
    lst = []
    for idx in range(len(hand)):
        temp = hand[idx]
        for jdx in range(idx+1, len(hand)):
            if hand[idx] == hand[jdx]:
                temp += hand[jdx]
        lst.append(temp)
    return max(lst)


def expected_value(held_dice, num_die_sides, num_free_dice):
    """
    Compute the expected value of the held_dice given that there
    are num_free_dice to be rolled, each with num_die_sides.

    held_dice: dice that you will hold
    num_die_sides: number of sides on each die
    num_free_dice: number of dice to be rolled

    Returns a floating point expected value
    """
    #dice_pos_val = [dummy for dummy in range(1, num_die_sides + 1)]
    #print dice_pos_val
    enum_out = gen_all_sequences([dummy for dummy in range(1, num_die_sides + 1)], num_free_dice)  
    total_score = 0.0
    ans = set()
    
    for idx in enum_out:
        temp = list(idx)
        for jdx in held_dice:
            temp.append(jdx)
        ans.add(tuple(temp))
        total_score += score(tuple(temp))     
    return total_score / len(ans)


def gen_all_holds(hand):
    """
    Generate all possible choices of dice from hand to hold.

    hand: full yahtzee hand

    Returns a set of tuples, where each tuple is dice to hold
    """
    choices = set([()])
    for idx in range(2**len(hand)):
        temp = []
        powerset = [idx & (2**dummy) > 0 for dummy in range(len(hand))]
        for jdx in range(len(powerset)):
            if powerset[jdx]:
                temp.append(hand[jdx])
        choices.add(tuple(temp))
    return choices



def strategy(hand, num_die_sides):
    """
    Compute the hold that maximizes the expected value when the
    discarded dice are rolled.

    hand: full yahtzee hand
    num_die_sides: number of sides on each die

    Returns a tuple where the first element is the expected score and
    the second element is a tuple of the dice to hold
    """
    
    possible_holds = list(gen_all_holds(hand))
    hand_val = float('-inf')
    hand_hold = []
    for item in possible_holds:
        exp_val = expected_value(item, num_die_sides, len(hand) - len(item))
        if exp_val > hand_val:
            hand_val = exp_val
            hand_hold = item
    return (hand_val, hand_hold)


def run_example():
    """
    Compute the dice to hold and expected score for an example hand
    """
    num_die_sides = 6
    hand = (1, 1, 1, 5, 6)
    hand_score, hold = strategy(hand, num_die_sides)
    print "Best strategy for hand", hand, "is to hold", hold, "with expected score", hand_score
    
    
run_example()

#import poc_holds_testsuite
#poc_holds_testsuite.run_suite(gen_all_holds)
                                       
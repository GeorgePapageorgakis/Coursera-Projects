"""
Student code for Word Wrangler game
"""

import urllib2
import codeskulptor
import poc_wrangler_provided as provided

WORDFILE = "assets_scrabble_words3.txt"


# Functions to manipulate ordered word lists

def remove_duplicates(list1):
    """
    Eliminate duplicates in a sorted list.

    Returns a new sorted list with the same elements in list1, but
    with no duplicates.

    This function can be iterative.
    """
    out = []
    for item in list1:
        if item not in out:
            out.append(item)
    return out

def intersect(list1, list2):
    """
    Compute the intersection of two sorted lists.

    Returns a new sorted list containing only elements that are in
    both list1 and list2.

    This function can be iterative.
    """
    out = []
    for item in list1:
        if item in list2 and item not in out:
            out.append(item)
    return out

# Functions to perform merge sort

def merge(list1, list2):
    """
    Merge two sorted lists.

    Returns a new sorted list containing all of the elements that
    are in either list1 and list2.

    This function can be iterative.
    """
    out = []
    list_a = list(list1)
    list_b = list(list2)
    while len(list_a) > 0 or len(list_b) > 0:
        if (len(list_a) > 0) and (len(list_b) > 0):
            if list_a[0] <= list_b[0]:
                out.append(list_a[0])
                list_a.remove(list_a[0])
            else:
                out.append(list_b[0])
                list_b.remove(list_b[0])
        elif len(list_a) > 0:
            out.append(list_a[0])
            list_a.remove(list_a[0])
        elif len(list_b) > 0:
            out.append(list_b[0])
            list_b.remove(list_b[0])
    return out
                
def merge_sort(list1):
    """
    Sort the elements of list1.

    Return a new sorted list with the same elements as list1.

    This function should be recursive.
    """
    if len(list1) <= 1:
        return list1
    left = list1[0 : len(list1)/2]
    right = list1[len(list1)/2 : len(list1)]
    left = merge_sort(left)
    right = merge_sort(right)    
    return merge(left, right)

# Function to generate all strings for the word wrangler game

def gen_all_strings(word):
    """
    Generate all strings that can be composed from the letters in word
    in any order.

    Returns a list of all strings that can be formed from the letters
    in word.

    This function should be recursive.
    """
    word = list(word)
    if len(word) <= 1:
        if '' not in word:
            word.append('')
        return word
    first = word.pop(0)
    rest_strings = gen_all_strings(word)
    rest = []
    for item in rest_strings:
        for idx in range(len(item) + 1):
            temp = item[0 : idx] + first + item[idx : len(item)]
            rest.append(temp)
    rest.extend(rest_strings)
    return rest

# Function to load words from a file

def load_words(filename):
    """
    Load word list from the file named filename.

    Returns a list of strings.
    """
    return []

def run():
    """
    Run game.
    """
    words = load_words(WORDFILE)
    wrangler = provided.WordWrangler(words, remove_duplicates, 
                                     intersect, merge_sort, 
                                     gen_all_strings)
    provided.run_game(wrangler)

# Uncomment when you are ready to try the game
run()

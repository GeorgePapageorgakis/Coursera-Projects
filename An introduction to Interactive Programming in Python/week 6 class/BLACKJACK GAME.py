# Mini-project #6 - Blackjack
import simplegui
import random
# load card sprite - 949x392 - source: jfitz.com
CARD_SIZE, CARD_CENTER = (73, 98), (36.5, 49)
card_images = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/cards.jfitz.png")

CARD_BACK_SIZE, CARD_BACK_CENTER = (71, 96), (35.5, 48)
card_back = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/card_back.png")    

# initialize some useful global variables
in_play, outcome, score, dealer, player, deck = False, "", 0, [], [], []

# define globals for cards
SUITS = ('C', 'S', 'H', 'D')
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
VALUES = {'A':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':10, 'Q':10, 'K':10}

# define card class
class Card:
    def __init__(self, suit, rank):
        if (suit in SUITS) and (rank in RANKS):
            self.suit = suit
            self.rank = rank
        else:
            self.suit = None
            self.rank = None
            print "Invalid card: ", suit, rank

    def __str__(self):
        return self.suit + self.rank

    def get_suit(self):
        return self.suit

    def get_rank(self):
        return self.rank

    def draw(self, canvas, pos):
        card_loc = (CARD_CENTER[0] + CARD_SIZE[0] * RANKS.index(self.rank), 
                    CARD_CENTER[1] + CARD_SIZE[1] * SUITS.index(self.suit))
        canvas.draw_image(card_images, card_loc, CARD_SIZE, [pos[0] + CARD_CENTER[0], pos[1] + CARD_CENTER[1]], CARD_SIZE)
        
# define hand class
class Hand:
    def __init__(self):
        self.hand_cards = [] # create Hand object

    def __str__(self):
        # return a string representation of a hand
        return "Hand contains " + " ".join(i.get_suit() + i.get_rank() for i in self.hand_cards)
    
    def add_card(self, card):
        self.hand_cards.append(card) # add a card object to a hand

    def get_value(self):
        value = 0
        ace = False
        # count aces as 1, if the hand has an ace, then add 10 to hand value if it doesn't bust
        for i in self.hand_cards:
            if not ace:
                if i.get_rank() == 'A' and value+10 < 21:
                    value += 10
                    ace = True
            # compute the value of the hand, see Blackjack video
            value += VALUES[i.get_rank()]
        return value

    def draw(self, canvas, p):
        # draw a hand on the canvas, use the draw method for cards 
        for i in range (1, len(self.hand_cards)):
            self.hand_cards[i].draw(canvas, [p[0] + CARD_CENTER[0] + i * (20 + CARD_SIZE[0]), p[1] + CARD_CENTER[1]])
         
       
# define deck class 
class Deck:  
    def __init__(self):
        self.deck_cards = []
        # create a Deck object
        for s in SUITS:
            for r in RANKS:
                self.deck_cards.append(Card(s, r))
        self.shuffle()

    def shuffle(self):
        # shuffle the deck 
        random.shuffle(self.deck_cards)

    def deal_card(self):
        return self.deck_cards.pop() # deal a card object from the deck
    
    def __str__(self):
        # return a string representing the deck   
        return "Deck contains " + " ".join(i.get_suit() + i.get_rank() for i in self.deck_cards)


#define event handlers for buttons
def deal():
    global outcome, in_play, dealer, player, deck, score   
    # deals the two cards to both the dealer and the player
    if in_play:
        outcome = 'You deal before end of round'
        current = 'New deal?'
        in_play = False
        score -= 1
    else:
        outcome = ""
        in_play = True
        dealer, player, deck = Hand(), Hand(), Deck()
        dealer.add_card(deck.deal_card())
        dealer.add_card(deck.deal_card())
        player.add_card(deck.deal_card())
        player.add_card(deck.deal_card())
        if player.get_value() == 21:
            stand()
    
def hit():
    global in_play, outcome, score, dealer, player
    # if the hand is in play, hit the player
    if in_play:
        player.add_card(deck.deal_card())      
        # if busted, assign a message to outcome, update in_play and score
        if player.get_value() > 21:            
            outcome = 'You went bust!'
            current = 'New deal?'
            in_play = False
            score -= 1
        elif player.get_value() == 21: # stand in case of blackjack   
            current = 'BlackJack'
            stand()
  
def stand():
    # if hand is in play, repeatedly hit dealer until his hand has value 17 or more
    global in_play, outcome, score, dealer, player
    
    if in_play:
        # assign a message to outcome, update in_play and score
        while dealer.get_value() < 17:
            dealer.add_card(deck.deal_card())
        if dealer.get_value() <= 21:
            if dealer.get_value() >= player.get_value():
                outcome = "You lose."
                score -= 1
            elif dealer.get_value() < player.get_value():
                outcome = "You win."
                score += 1
        else:
            outcome = "Dealer went busted and you won."
            score += 1
        in_play = False
    
# draw handler    
def draw(canvas):
    global in_play, dealer, player
    canvas.draw_polygon([(80, 200), (560, 200), (560, 330), (80, 330)], 4, 'Orange')
    canvas.draw_polygon([(80, 400), (560, 400), (560, 530), (80, 530)], 4, 'Orange')
    canvas.draw_text("Blackjack", [100, 100], 35, "Lime")
    canvas.draw_text("Score " + str(score), [450, 100], 25, "White")
    canvas.draw_text("Dealer", [80, 170], 25, "White")
    canvas.draw_text(outcome, [200, 170], 25, "White")   
    canvas.draw_text("Player", [80, 370], 25, "White")
    dealer.draw(canvas, [60, 170])
    if in_play:
        canvas.draw_image(card_back, CARD_BACK_CENTER, CARD_BACK_SIZE, [60 + CARD_BACK_SIZE[0], 170 + CARD_BACK_SIZE[1]], CARD_SIZE)
    else:
        dealer.hand_cards[0].draw(canvas, [60 + CARD_CENTER[0], 170 + CARD_CENTER[1]])
    if in_play:
        canvas.draw_text("Hit or stand?", [200, 370], 25, "White")
    else:
        canvas.draw_text("New deal?", [200, 370], 25, "White")   
    player.hand_cards[0].draw(canvas, [60 + CARD_CENTER[0], 370 + CARD_CENTER[1]])
    player.draw(canvas, [60, 370])

    
# initialization frame
frame = simplegui.create_frame("Blackjack", 600, 600)
frame.set_canvas_background("Green")


#create buttons and canvas callback
frame.add_button("Deal", deal, 200)
frame.add_button("Hit",  hit, 200)
frame.add_button("Stand", stand, 200)
frame.set_draw_handler(draw)

# get things rolling
deal()
frame.start()
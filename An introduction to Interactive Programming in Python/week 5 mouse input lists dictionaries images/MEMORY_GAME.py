# implementation of card game - Memory
import simplegui
import random
turns, state, index_x1, index_x2 = 0, 0, 0, 0

# helper function to initialize globals
def new_game():
    global cards, exposed, turns
    cards, exposed = [], []
    turns = 0
    #initialize cards
    for i in range(0, 8):
        cards.append(str(i))
        cards.append(str(i))
        exposed.append(0)
        exposed.append(0)
    #shuffle cards
    random.shuffle(cards)
    print cards
    if cards[0] != cards[0]:
        print cards[0]
        
# define event handlers
def mouseclick(pos):
    #add game state logic here
    global exposed,turns, state, index_x1, index_x2
    if exposed[pos[0]//50] == 0:
        if state == 0:
            state = 1
            index_x1 = pos[0]//50
            exposed[index_x1]=1    
        elif state == 1:
            state = 2
            index_x2 = pos[0]//50
            exposed[index_x2]=1  
        else:
            if state == 2 and (cards[index_x1] != cards[index_x2]):
                exposed[index_x1]=0
                exposed[index_x2]=0
            index_x1 = pos[0]//50
            exposed[index_x1]=1
            state = 1
            turns +=1        
                        
# cards are logically 50x100 pixels in size    
def draw(canvas):
    global cards
    label.set_text("Turns = "+ str(turns))
    for i in range(len(cards)):
        canvas.draw_text(cards[i], (i*50+6,75), 75, "White")
        if exposed[i] == 0:
            canvas.draw_polygon([[i*50, 0], [i*50+50, 0], [i*50+50, 100], [i*50, 100]], 5, 'Lime', 'Green') 

# create frame and add a button and labels
frame = simplegui.create_frame("Memory", 800, 100)
frame.add_button("Reset", new_game)
label = frame.add_label("Turns = 0")

# register event handlers
frame.set_mouseclick_handler(mouseclick)
frame.set_draw_handler(draw)

# get things rolling
new_game()
frame.start()

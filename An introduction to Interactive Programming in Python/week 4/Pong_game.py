# Implementation of classic arcade game Pong
import simplegui
import random

# initialize globals - pos and vel encode vertical info for paddles
WIDTH, HEIGHT = 600, 400       
BALL_RADIUS = 10
PAD_WIDTH, PAD_HEIGHT = 8, 80
HALF_PAD_WIDTH, HALF_PAD_HEIGHT = PAD_WIDTH / 2, PAD_HEIGHT / 2
#initialize paddles at middle height
paddle1_pos = [[0,0],[0,0]]
paddle2_pos = [[0,0],[0,0]]
paddle1_vel, paddle2_vel = 0, 0
score1, score2 = 0, 0
LEFT = False
RIGHT = True
ball_pos = [WIDTH / 2, HEIGHT / 2]
ball_vel = [0, 0]

# initialize ball_pos and ball_vel for new bal in middle of table
# if direction is RIGHT, the ball's velocity is upper right, else upper left
def spawn_ball(direction):
    #global ball_pos, ball_vel # these are vectors stored as lists
    if direction == LEFT:
        ball_vel[0] = random.randrange(-5, -1)
        ball_vel[1] = random.randrange(-5, 0)
    else:    
        ball_vel[0] = random.randrange(2, 5)
        ball_vel[1] = random.randrange(-5, 0)
       
def reset_ball():
    global ball_pos, ball_vel
    ball_pos = [WIDTH / 2, HEIGHT / 2]
    ball_vel = [0, 0]
  
def reset_paddles():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel
    paddle1_pos = [[HALF_PAD_WIDTH, HEIGHT//2 - HALF_PAD_HEIGHT],[HALF_PAD_WIDTH, HEIGHT//2 + HALF_PAD_HEIGHT]]
    paddle2_pos = [[WIDTH - HALF_PAD_WIDTH, HEIGHT//2 - HALF_PAD_HEIGHT],[WIDTH - HALF_PAD_WIDTH, HEIGHT//2 + HALF_PAD_HEIGHT]]
    paddle1_vel, paddle2_vel = 0, 0
    
# define event handlers
def new_game():
    global score1, score2  #these are ints
    reset_paddles()
    reset_ball()    
    spawn_ball(random.randrange(2))
    score1, score2 = 0, 0
    
def draw(canvas):
    global score1, score2, paddle1_pos, paddle2_pos, ball_pos, ball_vel, paddle1_vel, paddle2_vel
        
    # draw mid line and gutters
    canvas.draw_line([WIDTH / 2, 0],[WIDTH / 2, HEIGHT], 1, "White")
    canvas.draw_line([PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1, "White")
    canvas.draw_line([WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1, "White")
   
    # update ball
    ball_pos[0] += ball_vel[0]
    ball_pos[1] += ball_vel[1]
    
    #check if ball hits paddle if not reset ball
    if ball_pos[0] <= PAD_WIDTH + BALL_RADIUS:
        if ball_pos[1] >= paddle1_pos[0][1] and ball_pos[1] <= paddle1_pos[1][1]:
            ball_vel[0] = - ball_vel[0]
            if ball_vel[0] <= 20 and ball_vel[1] <= 20:
                ball_vel[0] *= 1.1
                ball_vel[1] *= 1.1
        else:
            #increase score
            score2 += 1
            reset_ball()
            spawn_ball(RIGHT)
    elif ball_pos[0] >= WIDTH - PAD_WIDTH - BALL_RADIUS:
        if ball_pos[1] >= paddle2_pos[0][1] and ball_pos[1] <= paddle2_pos[1][1]:
            ball_vel[0] = - ball_vel[0]
            if ball_vel[0] <= 20 and ball_vel[1] <= 20:
                ball_vel[0] *= 1.1
                ball_vel[1] *= 1.1
        else:
            #increase score
            score1 += 1
            reset_ball()
            spawn_ball(LEFT)        
    elif ball_pos[1] <= BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
    elif ball_pos[1] >= HEIGHT - 1 - BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
        
    # draw ball
    canvas.draw_circle(ball_pos, BALL_RADIUS, 3, "Red", "White") 
    
    # update paddle's vertical position, keep paddle on the screen
    if paddle1_vel != 0 or paddle2_vel != 0:
        if paddle1_pos[0][1] + paddle1_vel >= 0 and paddle1_pos[1][1] + paddle1_vel <= HEIGHT:
            paddle1_pos[0][1] += paddle1_vel
            paddle1_pos[1][1] += paddle1_vel
        if paddle2_pos[0][1] + paddle2_vel >= 0 and paddle2_pos[1][1] + paddle2_vel <= HEIGHT:
            paddle2_pos[0][1] += paddle2_vel
            paddle2_pos[1][1] += paddle2_vel

    # draw paddles
    canvas.draw_line(paddle1_pos[0], paddle1_pos[1], PAD_WIDTH, 'Red')
    canvas.draw_line(paddle2_pos[0], paddle2_pos[1], PAD_WIDTH, 'Red')
    
    # draw scores
    canvas.draw_text(str(score1), (WIDTH//4, HALF_PAD_HEIGHT), 36, 'white')
    canvas.draw_text(str(score2), (3*WIDTH//4, HALF_PAD_HEIGHT), 36, 'white')
    
def keydown(key):
    global paddle1_vel, paddle2_vel
    if key==simplegui.KEY_MAP["w"]:
        paddle1_vel = -5
    elif key==simplegui.KEY_MAP["s"]:
        paddle1_vel = 5
    elif key==simplegui.KEY_MAP["down"]:
        paddle2_vel = 5
    elif key==simplegui.KEY_MAP["up"]:
        paddle2_vel = -5      
    
def keyup(key):
    global paddle1_vel, paddle2_vel
    if key==simplegui.KEY_MAP["w"] or key==simplegui.KEY_MAP["s"]:
        paddle1_vel = 0
    elif key==simplegui.KEY_MAP["down"] or key==simplegui.KEY_MAP["up"]:
        paddle2_vel = 0

        
# create frame
frame = simplegui.create_frame("Pong", WIDTH, HEIGHT)
frame.add_button("Restart Game", new_game, 60)
frame.set_draw_handler(draw)
frame.set_keydown_handler(keydown)
frame.set_keyup_handler(keyup)

# start frame
new_game()
frame.start()

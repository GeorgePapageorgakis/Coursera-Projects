# template for "Stopwatch: The Game"
import simplegui

# define global variables
message, color = "0:00.0", "White"
width, height = 200, 200
interval = 100
time, total, success = 0, 0, 0

# define helper function format that converts time
# in tenths of seconds into formatted string A:BC.D
def format(t):
    global message
    seconds_minor, seconds_major, minutes = 0, 0, 0
    tens_seconds, seconds_minor, seconds_major, minutes = t%10, t%100//10, t%600//100, t//600
    return (str(minutes) + ":" + str(seconds_major) + str(seconds_minor) + "." + str(tens_seconds))
    
# define event handlers for buttons; "Start", "Stop", "Reset"
def Start():
    color_update("Lime")
    timer.start()
    
def Stop():
    global success, total, time
    if not timer.is_running():
        return
    total += 1
    if time%10 == 0:
        success += 1
    color_update("Red")
    timer.stop()
    
def Reset():
    global time, message, total, success
    timer.stop()
    time, total, success = 0, 0, 0
    color_update("White")
    message = format(0)

def color_update(c):
    global color
    color = c
        
# define event handler for timer with 0.1 sec interval
def update():
    global time, message
    time += 1
    message = format(time)

# define draw handler
def draw(canvas):
    canvas.draw_text(message, [width//4, height//1.8], 40, color)
    canvas.draw_text(str(success) + "/" + str(total), [width-40, 20], 24, "White")
    
# create frame
frame = simplegui.create_frame("Stopwatch: The Game", width, height)
frame.add_button('Start', Start)
frame.add_button('Stop', Stop)
frame.add_button('Reset', Reset)

# register event handlers
frame.set_draw_handler(draw)
timer = simplegui.create_timer(interval, update)


# start frame
frame.start()

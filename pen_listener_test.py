# pen_listener_test.py
from pynput import mouse
import time

def on_move(x, y):
    print("MOVE", x, y)

def on_click(x, y, button, pressed):
    print("CLICK", x, y, button, pressed)

print("Listening... move your pen / click for 10 seconds")
listener = mouse.Listener(on_move=on_move, on_click=on_click)
listener.start()
time.sleep(10)
listener.stop()
print("Done")
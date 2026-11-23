import tobii_research as tr
import csv
import time
import threading
import tkinter as tk
from pynput import mouse

# -----------------------------
# Globals
# -----------------------------
pen_x, pen_y = 0, 0
drawing = False
gaze_x, gaze_y = None, None

# -----------------------------
# Gaze Callback
# -----------------------------
def gaze_callback(gaze_data):
    global gaze_x, gaze_y

    left = gaze_data["left_gaze_point_on_display_area"]
    right = gaze_data["right_gaze_point_on_display_area"]

    # Use left eye if valid, else right
    if left[0] >= 0:
        gx, gy = left
    else:
        gx, gy = right

    # Convert relative (0–1) → screen approx (1920x1080 canvas)
    gaze_x = int(gx * 800)
    gaze_y = int(gy * 600)

# -----------------------------
# Pen Input via Mouse Capture
# -----------------------------
def on_move(x, y):
    global pen_x, pen_y
    pen_x, pen_y = x % 800, y % 600  # convert to window coordinates

def on_click(x, y, button, pressed):
    global drawing
    drawing = pressed

listener = mouse.Listener(on_move=on_move, on_click=on_click)
listener.start()

# -----------------------------
# Setup CSV Logging
# -----------------------------
csv_file = open("sync_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "pen_x", "pen_y", "gaze_x", "gaze_y"])

# -----------------------------
# GUI Setup (Tkinter Canvas)
# -----------------------------
win = tk.Tk()
win.title("Pen + Gaze Real-Time Sync Demo")

canvas = tk.Canvas(win, width=800, height=600, bg="white")
canvas.pack()

# -----------------------------
# Main Update Loop
# -----------------------------
def update():
    global pen_x, pen_y, gaze_x, gaze_y

    # Draw pen stroke
    if drawing:
        canvas.create_oval(pen_x-2, pen_y-2, pen_x+2, pen_y+2, fill="black")

    # Draw gaze point
    if gaze_x is not None:
        canvas.create_oval(gaze_x-5, gaze_y-5, gaze_x+5, gaze_y+5, fill="red")

    # Write to CSV (synchronized)
    csv_writer.writerow([
        int(time.time() * 1000),
        pen_x,
        pen_y,
        gaze_x,
        gaze_y
    ])

    win.after(10, update)   # update every 10ms

# -----------------------------
# Start Gaze Stream
# -----------------------------
trackers = tr.find_all_eyetrackers()
if len(trackers) == 0:
    print("No Tobii tracker found.")
    exit()

tracker = trackers[0]
print("Tracking with:", tracker)

tracker.subscribe_to(
    tr.EYETRACKER_GAZE_DATA,
    gaze_callback,
    as_dictionary=True
)

# -----------------------------
# Start Demo
# -----------------------------
update()
win.mainloop()

# Cleanup
tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)
csv_file.close()
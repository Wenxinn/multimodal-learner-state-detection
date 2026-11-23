# pen_logger.py
import time, csv
from pynput import mouse

outfile = open("pen_data.csv", "w", newline="", encoding="utf-8")
writer = csv.writer(outfile)
writer.writerow(["timestamp_ms", "x", "y", "button", "event"])

start = time.time()

def now_ms():
    return int((time.time() - start) * 1000)

def on_move(x, y):
    writer.writerow([now_ms(), x, y, "", "move"])

def on_click(x, y, button, pressed):
    writer.writerow([now_ms(), x, y, str(button), "down" if pressed else "up"])

with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
    print("Recording pen/mouse… press Ctrl+C in this window to stop.")
    try:
        listener.join()
    except KeyboardInterrupt:
        pass

outfile.close()
print("Saved pen data to pen_data.csv")
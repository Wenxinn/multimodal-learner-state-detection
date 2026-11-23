# pen_debug.py
import time

from gaze_pen_logger import GazePenLogger  # same module you use in main.py

logger = GazePenLogger()

def on_pen_event(event):
    print(
        f"t={event.timestamp_ms}, "
        f"x={event.pen_x}, y={event.pen_y}, "
        f"type={event.event_type}, pressure={event.pressure}"
    )

logger.pen_callback = on_pen_event  # if your class allows assigning a callback

print("Starting pen debug. Move the Wacom pen on the tablet, tap, etc.")
logger.start_logging("pen_debug.csv")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    logger.stop_logging()
    print("Stopped.")
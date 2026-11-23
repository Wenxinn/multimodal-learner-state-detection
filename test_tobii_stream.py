import tobii_research as tr
import time
import csv
from datetime import datetime

# 1. Find the eye tracker
found_eyetrackers = tr.find_all_eyetrackers()
if not found_eyetrackers:
    print("❌ No eye tracker found. Check USB & Tobii Eye Tracker Manager.")
    exit()

tracker = found_eyetrackers[0]
print("✅ Using eye tracker:", tracker.address)

# 2. Prepare CSV
OUT_FILE = "gaze_data.csv"

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_us", "left_x", "left_y", "right_x", "right_y"])

# 3. Callback to handle gaze events
def gaze_callback(gaze_data):
    left = gaze_data.get("left_gaze_point_on_display_area")
    right = gaze_data.get("right_gaze_point_on_display_area")
    ts = gaze_data.get("device_time_stamp")  # microseconds

    left_x, left_y = (left[0], left[1]) if left and left[0] is not None else (None, None)
    right_x, right_y = (right[0], right[1]) if right and right[0] is not None else (None, None)

    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, left_x, left_y, right_x, right_y])

# 4. Subscribe for ~30 seconds
print("👀 Starting gaze logging for about 30 seconds...")
tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_callback, as_dictionary=True)

try:
    time.sleep(30)  # log for 30 seconds while you do the task
finally:
    tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)
    print("✅ Gaze logging finished. Saved to gaze_data.csv")
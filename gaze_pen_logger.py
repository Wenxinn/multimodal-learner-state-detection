# gaze_pen_logger.py
# Logs pen (mouse) events with timestamps to a CSV file.
# Uses pynput, which you already installed in your venv.

import csv
import os
import time
from threading import Lock
from pynput import mouse


class GazePenLogger:
    def __init__(self):
        self.file = None
        self.writer = None
        self.listener = None
        self.lock = Lock()
        self.running = False

    def start(self, csv_path: str):
        """Start logging pen (mouse) movements to csv_path."""
        # Make sure folder exists
        folder = os.path.dirname(csv_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        # Open CSV and write header
        self.file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp_ms", "pen_x", "pen_y", "event_type", "button"])

        self.running = True

        def on_move(x, y):
            if not self.running:
                return
            with self.lock:
                self.writer.writerow([int(time.time() * 1000), x, y, "move", ""])

        def on_click(x, y, button, pressed):
            if not self.running:
                return
            event_type = "down" if pressed else "up"
            with self.lock:
                self.writer.writerow(
                    [int(time.time() * 1000), x, y, event_type, str(button)]
                )

        # Start background listener
        self.listener = mouse.Listener(on_move=on_move, on_click=on_click)
        self.listener.start()
        print(f"[GazePenLogger] Started logging to {csv_path}")

    def stop(self):
        """Stop logging and close file."""
        self.running = False

        if self.listener is not None:
            self.listener.stop()
            self.listener = None

        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

        print("[GazePenLogger] Stopped logging")


# This is what main.py imports:
logger = GazePenLogger()
"""
recorder.py
===========
Screen recording module for VideoCode.

Features:
  1. DPI awareness set at import via SetProcessDpiAwareness(2) before any Tk window exists.
  2. Fullscreen translucent region selector (drag to select, Esc cancels, returns even-dimension real pixels).
  3. Thread-safe mss capture instantiated inside the worker thread.
  4. Automatic BGRA to 3-channel BGR conversion with assertion.
  5. Precise 0.5s sampling using time.monotonic() to prevent time drift.
  6. Feeds frames to StabilityDetector and keeps ONLY the settled stable frames in memory.
  7. Black-frame detection to warn if DRM / video player blocks screen recording.
  8. Injectable grab_fn for automated tests and headless testing.
"""

import ctypes
import os
import sys
import threading
import time
import tkinter as tk
import cv2
import numpy as np

# 1. Set process DPI awareness before any Tk window is initialized
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from videocode_core import StabilityDetector


class ScreenRecorder:
    """
    Manages capturing a screen region on a background thread at fixed intervals (default 0.5s),
    filtering each frame through StabilityDetector in real time.
    """

    def __init__(self, sample_interval=0.5, grab_fn=None, time_fn=None):
        self.sample_interval = sample_interval
        self.grab_fn = grab_fn  # Optional injectable callable: grab_fn(region) -> BGR or BGRA frame
        self.time_fn = time_fn or time.monotonic
        self._thread = None
        self._stop_event = threading.Event()
        self.region = None
        self.stable_frames = []
        self.stats = {}
        self.black_frame_warning = False
        self.is_recording = False

    def start(self, region):
        """
        Start capturing `region` = (x1, y1, x2, y2) in a background thread.
        """
        if self.is_recording:
            raise RuntimeError("Recorder is already running.")

        x1, y1, x2, y2 = region
        # Ensure coordinates are properly ordered
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        # Ensure even width and height
        if (x2 - x1) % 2 != 0:
            x2 += 1
        if (y2 - y1) % 2 != 0:
            y2 += 1

        self.region = (x1, y1, x2, y2)
        self.stable_frames = []
        self.stats = {}
        self.black_frame_warning = False
        self._stop_event.clear()
        self.is_recording = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop recording, finish stability detection, and return (stable_frames, stats).
        stable_frames: list of (time_seconds, frame_bgr)
        """
        if not self.is_recording:
            return self.stable_frames, self.stats

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self.is_recording = False
        return self.stable_frames, self.stats

    def _capture_loop(self):
        x1, y1, x2, y2 = self.region
        w, h = x2 - x1, y2 - y1

        # mss instance MUST be created inside the worker thread (not thread-safe on Windows)
        sct = None
        if self.grab_fn is None:
            import mss
            sct_cls = getattr(mss, "MSS", mss.mss)
            sct = sct_cls()
            monitor = {"top": y1, "left": x1, "width": w, "height": h}

        detector = StabilityDetector(min_stable=2)
        kept_frames = []
        total_samples = 0
        black_samples = 0

        start_time = self.time_fn()
        tick_idx = 0

        try:
            while not self._stop_event.is_set():
                now = self.time_fn()
                elapsed = now - start_time

                # Grab frame
                if self.grab_fn is not None:
                    raw = self.grab_fn(self.region)
                    if raw is None:
                        break
                    if raw.ndim == 3 and raw.shape[2] == 4:
                        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                    else:
                        frame = raw
                else:
                    sct_img = sct.grab(monitor)
                    # Convert mss 4-channel BGRA to 3-channel BGR
                    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

                # Channel check assertion
                assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected 3 channels (BGR), got {frame.shape}"

                total_samples += 1
                if frame.max() < 10:
                    black_samples += 1

                # Feed to stability detector
                res = detector.feed(elapsed, frame)
                if res:
                    (t, f), _ = res
                    kept_frames.append((t, f))

                # Schedule next tick without drift
                tick_idx += 1
                next_tick = start_time + (tick_idx * self.sample_interval)
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    if self._stop_event.wait(timeout=sleep_time):
                        break

            # Stream finished: final flush
            final_res = detector.finish()
            if final_res:
                (t, f), _ = final_res
                kept_frames.append((t, f))

        finally:
            if sct is not None:
                sct.close()

        black_ratio = (black_samples / total_samples) if total_samples > 0 else 0.0
        self.black_frame_warning = (total_samples >= 3 and black_ratio > 0.9)
        self.stable_frames = kept_frames
        self.stats = {
            "sampled": total_samples,
            "stable_moments": len(kept_frames),
            "black_samples": black_samples,
            "black_ratio": round(black_ratio, 2),
            "black_frame_warning": self.black_frame_warning,
        }


# ---------------------------------------------------------------------------
# Fullscreen Translucent Region Selector
# ---------------------------------------------------------------------------
def select_region(parent=None):
    """
    Displays a fullscreen translucent overlay allowing the user to drag a rectangle.
    Pressing Escape cancels and returns None.
    Releasing mouse confirms. Returns (x1, y1, x2, y2) in real pixel coordinates.
    Boxes smaller than 20x20 are rejected.
    """
    created_tk = False
    if parent is None:
        # Check if a default root exists
        if tk._default_root is None:
            root = tk.Tk()
            root.withdraw()
            created_tk = True
        else:
            root = tk._default_root
    else:
        root = parent

    overlay = tk.Toplevel(root)
    overlay.title("Select Area - VideoCode")
    overlay.attributes("-fullscreen", True)
    overlay.attributes("-alpha", 0.3)
    overlay.attributes("-topmost", True)
    overlay.configure(bg="black")
    overlay.config(cursor="cross")

    canvas = tk.Canvas(overlay, cursor="cross", bg="gray15", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Instructions text
    canvas.create_text(
        overlay.winfo_screenwidth() // 2,
        40,
        text="Click and drag to select code area. Press ESC to cancel.",
        fill="white",
        font=("Segoe UI", 16, "bold"),
    )

    selection = {"start_x": 0, "start_y": 0, "rect_id": None, "result": None}

    def on_press(event):
        selection["start_x"] = event.x_root
        selection["start_y"] = event.y_root
        if selection["rect_id"] is not None:
            canvas.delete(selection["rect_id"])
        selection["rect_id"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2
        )

    def on_drag(event):
        if selection["rect_id"] is not None:
            # Map root coords back to canvas
            x0 = canvas.canvasx(min(selection["start_x"], event.x_root))
            y0 = canvas.canvasy(min(selection["start_y"], event.y_root))
            x1 = canvas.canvasx(max(selection["start_x"], event.x_root))
            y1 = canvas.canvasy(max(selection["start_y"], event.y_root))
            canvas.coords(selection["rect_id"], x0, y0, x1, y1)

    def on_release(event):
        x1, x2 = min(selection["start_x"], event.x_root), max(selection["start_x"], event.x_root)
        y1, y2 = min(selection["start_y"], event.y_root), max(selection["start_y"], event.y_root)

        # Reject boxes that are too small
        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            # Ensure even dimensions
            if (x2 - x1) % 2 != 0:
                x2 += 1
            if (y2 - y1) % 2 != 0:
                y2 += 1
            selection["result"] = (x1, y1, x2, y2)
        else:
            selection["result"] = None
        overlay.destroy()

    def on_cancel(event):
        selection["result"] = None
        overlay.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    overlay.bind("<Escape>", on_cancel)

    # Focus and wait
    overlay.focus_set()
    overlay.wait_window()

    if created_tk:
        root.destroy()

    return selection["result"]


# Module-level convenience instance
_default_recorder = None


def start_recording(region, grab_fn=None):
    global _default_recorder
    _default_recorder = ScreenRecorder(grab_fn=grab_fn)
    _default_recorder.start(region)
    return _default_recorder


def stop_recording():
    global _default_recorder
    if _default_recorder is None:
        return [], {}
    res = _default_recorder.stop()
    _default_recorder = None
    return res


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------
def demo():
    print("=" * 60)
    print("  VideoCode Recorder Demo")
    print("  Please select the code area on your screen ...")
    print("=" * 60)

    region = select_region()
    if not region:
        print("Selection cancelled or too small. Exiting.")
        return

    print(f"Selected region: {region} (width: {region[2]-region[0]}, height: {region[3]-region[1]})")
    print("Recording for 10 seconds (sampling every 0.5s) ...")

    rec = ScreenRecorder(sample_interval=0.5)
    rec.start(region)

    for i in range(10, 0, -1):
        print(f"  {i}s remaining...", end="\r", flush=True)
        time.sleep(1.0)
    print("\nStopping recorder...")

    frames, stats = rec.stop()
    print(f"\nResults:")
    print(f"  Total samples taken:  {stats.get('sampled', 0)}")
    print(f"  Stable moments kept:  {stats.get('stable_moments', 0)}")
    print(f"  Black frame warning:  {stats.get('black_frame_warning', False)}")

    # Save kept frames into sessions/demo/
    demo_dir = os.path.join(os.path.dirname(__file__), "sessions", "demo")
    os.makedirs(demo_dir, exist_ok=True)

    for idx, (t, f) in enumerate(frames, 1):
        filename = f"frame_{idx:02d}_{t:05.1f}s.png"
        filepath = os.path.join(demo_dir, filename)
        cv2.imwrite(filepath, f)

    print(f"\nSaved {len(frames)} stable frame PNGs to: {demo_dir}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        print("Usage: python recorder.py --demo")

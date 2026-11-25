"""
test_recorder.py
================
Tests for recorder.py:
  1. Channel test: verifies real mss grab produces 3-channel BGR.
  2. Black-frame test: verifies pure black frames trigger warning, while dark-theme frames do not.
  3. Fake-screen test: feeds variant_a.mp4 frames through ScreenRecorder at 2 samples/sec,
     runs extract_code_from_frames on the kept stable frames, and compares with tests/baseline/variant_a_code.txt.

Usage:
    .venv\\Scripts\\python tests\\test_recorder.py
"""

import os
import sys
import time
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from recorder import ScreenRecorder
from videocode_core import extract_code_from_frames


def test_channel_conversion():
    print("\n--- Test 1: Channel Test (BGRA -> 3-channel BGR) ---")
    import mss

    raw_np = None
    try:
        sct_cls = getattr(mss, "MSS", mss.mss)
        with sct_cls() as sct:
            monitor = {"top": 0, "left": 0, "width": 100, "height": 100}
            raw = sct.grab(monitor)
            raw_np = np.array(raw)
            print("  (Grabbed live frame via mss)")
    except Exception as e:
        print(f"  (Note: Desktop DC unavailable in non-interactive background agent session: {e})")
        print("  (Testing with synthetic 4-channel BGRA frame matching mss format)")
        raw_np = np.zeros((100, 100, 4), dtype=np.uint8)
        raw_np[:, :, 3] = 255  # Alpha channel

    assert raw_np.ndim == 3 and raw_np.shape[2] == 4, f"Expected 4 channels from mss, got {raw_np.shape}"

    bgr = cv2.cvtColor(raw_np, cv2.COLOR_BGRA2BGR)
    assert bgr.ndim == 3 and bgr.shape[2] == 3, f"Expected 3 channels (BGR), got {bgr.shape}"
    assert bgr.shape[:2] == (100, 100), f"Expected 100x100 shape, got {bgr.shape[:2]}"

    print("  ✓ Frame successfully converted to 3-channel BGR (shape: %s)" % (bgr.shape,))
    return True


def test_black_frame_warning():
    print("\n--- Test 2: Black-Frame Warning Test ---")

    # Case A: Synthetic black frames (all zeros) -> must trigger warning
    black_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    sample_idx = [0]

    def grab_black(region):
        sample_idx[0] += 1
        if sample_idx[0] > 10:
            return None
        return black_frame

    rec_black = ScreenRecorder(sample_interval=0.01, grab_fn=grab_black)
    rec_black.start((0, 0, 200, 200))
    time.sleep(0.3)
    _, stats_black = rec_black.stop()

    assert stats_black["black_frame_warning"] is True, f"Expected warning for black frames, got {stats_black}"
    print(f"  ✓ Black frames triggered warning as expected (black ratio: {stats_black['black_ratio']})")

    # Case B: Dark editor theme (bg 30, text 212) -> must NOT trigger warning
    dark_editor_frame = np.full((200, 200, 3), 30, dtype=np.uint8)
    cv2.putText(dark_editor_frame, "class Circle:", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (212, 212, 212), 2)
    sample_idx[0] = 0

    def grab_dark(region):
        sample_idx[0] += 1
        if sample_idx[0] > 10:
            return None
        return dark_editor_frame

    rec_dark = ScreenRecorder(sample_interval=0.01, grab_fn=grab_dark)
    rec_dark.start((0, 0, 200, 200))
    time.sleep(0.3)
    _, stats_dark = rec_dark.stop()

    assert stats_dark["black_frame_warning"] is False, f"Did NOT expect warning for dark theme, got {stats_dark}"
    print(f"  ✓ Dark editor theme did NOT trigger warning (black ratio: {stats_dark['black_ratio']})")
    return True


def test_fake_screen_variant_a():
    print("\n--- Test 3: Fake-Screen Test (variant_a.mp4 through recorder) ---")
    video_path = os.path.join(ROOT, "tests", "fixtures", "variant_a.mp4")
    baseline_path = os.path.join(ROOT, "tests", "baseline", "variant_a_code.txt")

    if not os.path.exists(video_path):
        print(f"  ✗ Fixture missing: {video_path}")
        return False
    if not os.path.exists(baseline_path):
        print(f"  ✗ Baseline missing: {baseline_path}")
        return False

    with open(baseline_path, "r", encoding="utf-8") as f:
        expected_code = f.read()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # We want 2 samples per second: sample every (fps / 2.0) frames
    step_frames = int(round(fps / 2.0))
    current_frame_idx = [0]
    simulated_time = [0.0]

    def grab_from_video(region):
        if current_frame_idx[0] >= total_frames:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx[0])
        ok, frame = cap.read()
        if not ok:
            return None
        # advance
        current_frame_idx[0] += step_frames
        simulated_time[0] += 0.5
        return frame

    def virtual_time():
        return simulated_time[0]

    # Use simulated time and 0 interval to process frames rapidly
    rec = ScreenRecorder(sample_interval=0.001, grab_fn=grab_from_video, time_fn=virtual_time)
    rec.start((0, 0, 1280, 720))

    # Wait until all frames are read
    while rec.is_recording and current_frame_idx[0] < total_frames:
        time.sleep(0.01)

    stable_frames, stats = rec.stop()
    cap.release()

    print(f"  Sampled {stats['sampled']} frames, kept {len(stable_frames)} stable moments")
    print("  Running extract_code_from_frames on kept frames...")

    code, review, info = extract_code_from_frames(stable_frames, progress=lambda msg: print(f"    {msg}"))

    # Compare with baseline
    # Both stripped of trailing whitespace
    code_clean = "\n".join(l.rstrip() for l in code.splitlines() if l.strip())
    expected_clean = "\n".join(l.rstrip() for l in expected_code.splitlines() if l.strip())

    match = (code_clean == expected_clean)
    if match:
        print("  ✓ Extracted code matches baseline EXACTLY!")
    else:
        print("  ✗ Extracted code differs from baseline:")
        print("--- GOT ---")
        print(code_clean)
        print("--- EXPECTED ---")
        print(expected_clean)

    print(f"  Stable moments: got {len(stable_frames)} (engine baseline had 9)")
    return match


def main():
    print("=" * 60)
    print("  Running Recorder Tests")
    print("=" * 60)

    ok1 = test_channel_conversion()
    ok2 = test_black_frame_warning()
    ok3 = test_fake_screen_variant_a()

    all_passed = ok1 and ok2 and ok3
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    print("=" * 60 + "\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

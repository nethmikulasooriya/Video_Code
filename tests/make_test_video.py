"""
make_test_video.py
==================
Generates synthetic tutorial videos with known ground-truth Python code.
Run once to create the test fixtures:

    .venv\\Scripts\\python tests\\make_test_video.py [a|b|c|d|all]

Output files (all inside tests/fixtures/):
    variant_a.mp4   - 1280x720,  7-line Python snippet
    variant_b.mp4   - 854x480,   same snippet downscaled
    variant_c.mp4   - 1280x720,  22-line program (17 non-blank) in a 12-line scrolling window
    variant_d.mp4   - 1280x720,  stress test: 18px font, short variable names, digits 0 & 8, mixed-case
    ground_truth_a.txt
    ground_truth_b.txt  (identical to a)
    ground_truth_c.txt
    ground_truth_d.txt
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\consola.ttf"
if not os.path.exists(FONT_PATH):
    for name in ["cour.ttf", "lucon.ttf", "DejaVuSansMono.ttf"]:
        p = os.path.join(r"C:\Windows\Fonts", name)
        if os.path.exists(p):
            FONT_PATH = p
            break
    else:
        sys.exit("ERROR: No monospace font found in C:\\Windows\\Fonts")

# ---------------------------------------------------------------------------
# Colour palette (VS Code Dark+)
# ---------------------------------------------------------------------------
BG    = (30,  30,  30)   # #1E1E1E – editor background
FG    = (212, 212, 212)  # #D4D4D4 – default text

# ---------------------------------------------------------------------------
# Ground-truth code
# ---------------------------------------------------------------------------
# Variant A/B: 7 non-blank lines
CODE_AB_LINES = [
    "class Circle:",
    "    def __init__(self, radius):",
    "        self.radius = radius",
    "    def area(self):",
    "        return 3.14159 * self.radius ** 2",
    "    def __str__(self):",
    '        return f"Circle(r={self.radius})"',
]

# Variant C: 22 total lines, 17 non-blank (5 blank separators between sections).
CODE_C_LINES = [
    "import math",
    "",
    "class Shape:",
    "    def area(self):",
    "        return None",
    "",
    "class Circle(Shape):",
    "    def __init__(self, radius):",
    "        self.radius = radius",
    "    def area(self):",
    "        return math.pi * self.radius ** 2",
    "",
    "class Rectangle(Shape):",
    "    def __init__(self, width, height):",
    "        self.width = width",
    "        self.height = height",
    "    def area(self):",
    "        return self.width * self.height",
    "",
    "def print_area(shape):",
    '    print(f"Area: {shape.area():.2f}")',
    "",
]

# Variant D (Stress test):
# 18px font, short variable names (x = 10, i, n, self.r = r), digits 0 and 8, mixed-case names (Circle)
CODE_D_LINES = [
    "class Circle:",
    "    def __init__(self, r):",
    "        self.r = r",
    "        x = 10",
    "        n = 80",
    "    def calc(self, i):",
    "        return self.r * i + 0",
    "    def __str__(self):",
    '        return f"Circle(r={self.r})"',
]

# ---------------------------------------------------------------------------
# Video parameters
# ---------------------------------------------------------------------------
FPS          = 10   # frames per second
FONT_SIZE_PX = 20   # 20 px default for A, B, C
PAD_LEFT     = 50   # left margin (column-zero x)
PAD_TOP      = 40   # top margin
LINE_H       = 26   # pixel height per code line (matches 20 px font)
CURSOR_W     = 2    # cursor width in pixels
BLINK_HZ     = 1    # cursor toggles every BLINK_HZ * FPS frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_font(size=FONT_SIZE_PX):
    return ImageFont.truetype(FONT_PATH, size)


def _add_noise(frame, amount=2):
    """Very mild salt-and-pepper noise."""
    noise = np.random.randint(-amount, amount + 1, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_blank(w, h):
    """BGR numpy array filled with the editor background colour."""
    return np.full((h, w, 3), BG, dtype=np.uint8)


def _render_lines(frame, lines, font, caret_col=None, caret_row=None, cursor_on=True, line_h=LINE_H):
    """
    Draw `lines` (list of str) onto `frame`.
    Returns a new numpy BGR array.
    """
    h, w = frame.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for row_idx, line in enumerate(lines):
        y = PAD_TOP + row_idx * line_h
        if y + line_h > h:
            break
        if line:
            draw.text((PAD_LEFT, y), line, font=font, fill=tuple(reversed(FG)))

    # Blinking cursor
    if cursor_on and caret_row is not None and caret_col is not None:
        cur_line = lines[caret_row] if caret_row < len(lines) else ""
        text_so_far = cur_line[:caret_col]
        bbox = draw.textbbox((0, 0), text_so_far, font=font)
        cx = PAD_LEFT + bbox[2]
        cy = PAD_TOP + caret_row * line_h
        draw.rectangle([cx, cy, cx + CURSOR_W, cy + line_h - 2],
                       fill=tuple(reversed(FG)))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _write_video(path, frames, fps, w, h):
    w = w if w % 2 == 0 else w + 1
    h = h if h % 2 == 0 else h + 1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(cv2.resize(f, (w, h)))
    out.release()
    print(f"  wrote {len(frames)} frames -> {path}")


# ---------------------------------------------------------------------------
# Video A/B builder
# ---------------------------------------------------------------------------

def build_video_ab(w=1280, h=720):
    font = load_font(FONT_SIZE_PX)
    frames = []
    typed = [""]
    cursor_row = 0

    def hold(n_frames):
        for i in range(n_frames):
            on = ((len(frames) + i) // BLINK_HZ) % 2 == 0
            blank = _make_blank(w, h)
            rendered = _render_lines(blank, typed, font,
                                     caret_row=cursor_row,
                                     caret_col=len(typed[cursor_row]),
                                     cursor_on=on, line_h=LINE_H)
            frames.append(_add_noise(rendered))

    for line_idx, line in enumerate(CODE_AB_LINES):
        typed[cursor_row] = ""
        for ch in line:
            typed[cursor_row] += ch
            hold(1)
        hold(FPS * 2)
        if line_idx < len(CODE_AB_LINES) - 1:
            typed.append("")
            cursor_row += 1

    hold(FPS)
    typed[4] = "        return 3.14159 * self.radius ** 3"
    hold(FPS * 2)
    typed[4] = "        return 3.14159 * self.radius ** 2"
    hold(FPS * 2)

    del typed[6]
    del typed[5]
    cursor_row = min(cursor_row, len(typed) - 1)
    hold(FPS * 2)
    typed.insert(5, "    def __str__(self):")
    typed.insert(6, '        return f"Circle(r={self.radius})"')
    cursor_row = 6
    hold(FPS * 3)

    return frames


# ---------------------------------------------------------------------------
# Video C builder (12-line scrolling window over 22-line program)
# ---------------------------------------------------------------------------

def build_video_c(w=1280, h=720, window_lines=12):
    font = load_font(FONT_SIZE_PX)
    frames = []
    all_lines = list(CODE_C_LINES)
    visible_start = 0

    def current_window():
        return all_lines[visible_start: visible_start + window_lines]

    def hold(n_frames, caret_row_in_window, caret_col):
        for i in range(n_frames):
            on = ((len(frames) + i) // BLINK_HZ) % 2 == 0
            blank = _make_blank(w, h)
            rendered = _render_lines(blank, current_window(), font,
                                     caret_row=caret_row_in_window,
                                     caret_col=caret_col,
                                     cursor_on=on, line_h=LINE_H)
            frames.append(_add_noise(rendered))

    def cur_row_in_win(line_idx):
        nonlocal visible_start
        row = line_idx - visible_start
        if row >= window_lines:
            visible_start = line_idx - window_lines + 1
            row = window_lines - 1
        return row

    for line_idx, line in enumerate(all_lines):
        if not line:
            row = cur_row_in_win(line_idx)
            hold(FPS // 2, row, 0)
            continue

        for char_count in range(1, len(line) + 1):
            row = cur_row_in_win(line_idx)
            hold(1, row, char_count)

        row = cur_row_in_win(line_idx)
        hold(FPS * 2, row, len(line))

    hold(FPS * 3, cur_row_in_win(len(all_lines) - 1), 0)
    return frames


# ---------------------------------------------------------------------------
# Video D builder (Stress test: 18px font, short variable names, digits 0 & 8)
# ---------------------------------------------------------------------------

def build_video_d(w=1280, h=720):
    font_size = 18
    line_h = 24
    font = load_font(font_size)
    frames = []
    typed = [""]
    cursor_row = 0

    def hold(n_frames):
        for i in range(n_frames):
            on = ((len(frames) + i) // BLINK_HZ) % 2 == 0
            blank = _make_blank(w, h)
            rendered = _render_lines(blank, typed, font,
                                     caret_row=cursor_row,
                                     caret_col=len(typed[cursor_row]),
                                     cursor_on=on, line_h=line_h)
            frames.append(_add_noise(rendered))

    for line_idx, line in enumerate(CODE_D_LINES):
        typed[cursor_row] = ""
        for ch in line:
            typed[cursor_row] += ch
            hold(1)
        hold(FPS * 2)
        if line_idx < len(CODE_D_LINES) - 1:
            typed.append("")
            cursor_row += 1

    hold(FPS * 3)
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if target in ("all", "a", "b"):
        if target in ("all", "a") or not os.path.exists(os.path.join(FIXTURES, "variant_a.mp4")):
            print("Building variant A (1280x720) ...")
            frames_a = build_video_ab(1280, 720)
            _write_video(os.path.join(FIXTURES, "variant_a.mp4"), frames_a, FPS, 1280, 720)
        else:
            frames_a = None

        if target in ("all", "b") or not os.path.exists(os.path.join(FIXTURES, "variant_b.mp4")):
            print("Building variant B (854x480, downscaled from A) ...")
            if frames_a is None:
                frames_a = build_video_ab(1280, 720)
            frames_b = [cv2.resize(f, (854, 480)) for f in frames_a]
            _write_video(os.path.join(FIXTURES, "variant_b.mp4"), frames_b, FPS, 854, 480)

    if target in ("all", "c"):
        print("Building variant C (1280x720, scrolling 12-line window over 22-line program) ...")
        frames_c = build_video_c(1280, 720, window_lines=12)
        _write_video(os.path.join(FIXTURES, "variant_c.mp4"), frames_c, FPS, 1280, 720)

    if target in ("all", "d"):
        print("Building variant D (1280x720, stress test: 18px font, short vars, digits 0 & 8) ...")
        frames_d = build_video_d(1280, 720)
        _write_video(os.path.join(FIXTURES, "variant_d.mp4"), frames_d, FPS, 1280, 720)

    # Write ground truth files
    gt_ab = "\n".join(CODE_AB_LINES)
    gt_c  = "\n".join(CODE_C_LINES)
    gt_d  = "\n".join(CODE_D_LINES)

    for name, text in [("ground_truth_a.txt", gt_ab),
                       ("ground_truth_b.txt", gt_ab),
                       ("ground_truth_c.txt", gt_c),
                       ("ground_truth_d.txt", gt_d)]:
        p = os.path.join(FIXTURES, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  wrote {p}")

    print("All requested fixtures updated in tests/fixtures/")


if __name__ == "__main__":
    main()

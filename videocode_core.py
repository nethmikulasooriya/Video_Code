import re
import statistics
from collections import Counter
from difflib import SequenceMatcher

import cv2
import numpy as np
from rapidfuzz import fuzz


# ---------------------------------------------------------------- helpers
def to_seconds(t):
    """'2:15' -> 135.0, '1:02:03' -> 3723.0, 90 -> 90.0"""
    if isinstance(t, (int, float)):
        return float(t)
    total = 0.0
    for part in str(t).split(":"):
        total = total * 60 + float(part)
    return total


def video_info(video):
    cap = cv2.VideoCapture(video)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frames"] / info["fps"]
    cap.release()
    return info


def get_frame(video, t, region=None):
    """One frame at time t (seconds). Used for choosing the code box."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, to_seconds(t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Could not read a frame at that time")
    if region:
        x1, y1, x2, y2 = region
        frame = frame[y1:y2, x1:x2]
    return frame


# ------------------------------------------------ step 1-3: find stable frames
def _thumb(img, width=640):
    h, w = img.shape[:2]
    width = min(width, w)
    small = cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _changed(a, b, pixel_thresh=40):
    """
    True if something text-sized changed between two thumbnails.
    A blinking cursor is a thin 1-2 px bar, so it is ignored; a typed character is a blob.
    """
    if a is None or b is None or a.shape != b.shape:
        return True
    mask = (cv2.absdiff(a, b) > pixel_thresh).astype(np.uint8)
    if mask.sum() < 8:
        return False
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    return any(st[i, cv2.CC_STAT_WIDTH] >= 7 and st[i, cv2.CC_STAT_HEIGHT] >= 7 for i in range(1, n))


def _sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


class StabilityDetector:
    """
    Online detector that filters raw frames down to stable moments.
    Same logic and thresholds as stable_frames():
    - min_stable consecutive samples required before settling
    - change detection ignores thin cursor-like changes
    - keeps the sharpest frame from the recent samples
    - skips duplicates of the last saved state
    """

    def __init__(self, min_stable=2):
        self.min_stable = min_stable
        self.stats = {"sampled": 0, "stable_moments": 0}
        self.prev_thumb = None
        self.last_saved_thumb = None
        self.run = []
        self.run_len = 0

    def _flush(self, force=False):
        if self.run_len >= (1 if force else self.min_stable) and self.run:
            best = max(self.run[-3:], key=lambda r: r[0])  # sharpest of the last few
            th = _thumb(best[2])
            if _changed(th, self.last_saved_thumb):
                self.last_saved_thumb = th
                self.stats["stable_moments"] += 1
                return (best[1], best[2])
        return None

    def feed(self, time_seconds, frame):
        """
        Feed one sampled frame at time_seconds.
        Returns (out, stats) if a settled stable moment was completed, else None.
        out is (time_seconds, frame).
        """
        self.stats["sampled"] += 1
        th = _thumb(frame)
        out = None
        if _changed(self.prev_thumb, th):
            out = self._flush()
            self.run, self.run_len = [], 0
        self.run.append((_sharpness(frame), time_seconds, frame))
        self.run = self.run[-3:]
        self.run_len += 1
        self.prev_thumb = th
        if out:
            return out, self.stats
        return None

    def finish(self):
        """
        Final flush at the end of stream.
        Returns (out, stats) if a remaining stable moment was completed, else None.
        """
        out = self._flush(force=True)
        if out:
            return out, self.stats
        return None


def stable_frames(video, start, end, region=None, samples_per_sec=2.0, min_stable=2):
    """
    Walk through the video and yield (time, frame), stats for each moment where the
    code stayed unchanged for at least `min_stable` samples (the screen 'settled').
    Duplicates of the previous saved state are skipped.
    """
    start, end = to_seconds(start), to_seconds(end)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f, end_f = int(start * fps), min(int(end * fps), total - 1)
    step = max(1, int(round(fps / samples_per_sec)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    detector = StabilityDetector(min_stable=min_stable)

    idx = start_f
    while idx <= end_f:
        if (idx - start_f) % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            if region:
                x1, y1, x2, y2 = region
                frame = frame[y1:y2, x1:x2]
            res = detector.feed(idx / fps, frame)
            if res:
                yield res
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()
    res = detector.finish()
    if res:
        yield res


# ------------------------------------------------------------ step 4: OCR
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def _ocr_rapid(frame, scale=2.0):
    big = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result, _ = get_engine()(big)
    items = []
    for box, text, score in (result or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({
            "x1": min(xs) / scale, "x2": max(xs) / scale,
            "yc": (min(ys) + max(ys)) / 2 / scale, "h": (max(ys) - min(ys)) / scale,
            "text": text.rstrip(), "conf": float(score),
        })
    return items


def _prep_for_tesseract(frame, scale):
    """Dark theme -> light text becomes dark text on white, which Tesseract reads best."""
    dark = frame.mean() < 127
    gray = frame.max(axis=2) if dark else frame.min(axis=2)  # keeps coloured syntax visible
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if dark:
        gray = cv2.bitwise_not(gray)
    return cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _ocr_tesseract(frame, scale=2.0):
    import pytesseract
    img = _prep_for_tesseract(frame, scale)
    d = pytesseract.image_to_data(
        img, config="--oem 1 --psm 6 -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DICT)
    items = []
    for i, text in enumerate(d["text"]):
        if not text.strip() or float(d["conf"][i]) < 0:
            continue
        x, y, w, h = d["left"][i] - 20, d["top"][i] - 20, d["width"][i], d["height"][i]
        items.append({
            "x1": x / scale, "x2": (x + w) / scale, "yc": (y + h / 2) / scale, "h": h / scale,
            "text": text, "conf": float(d["conf"][i]) / 100,
            "row": (d["block_num"][i], d["par_num"][i], d["line_num"][i]),
        })
    return items


def ocr_frame(frame, engine="tesseract", scale=2.0):
    """Returns word/line boxes in original pixels: x1, x2, yc, h, text, conf."""
    return _ocr_tesseract(frame, scale) if engine == "tesseract" else _ocr_rapid(frame, scale)


# ----------------------------------------------- step 5: boxes -> code lines
def group_lines(items):
    """Group boxes that share a row into rows (top to bottom)."""
    if not items:
        return []
    if all("row" in i for i in items):
        rows = {}
        for it in items:
            rows.setdefault(it["row"], []).append(it)
        rows = sorted(rows.values(), key=lambda r: statistics.mean(i["yc"] for i in r))
        return [sorted(r, key=lambda i: i["x1"]) for r in rows]
    med_h = statistics.median(i["h"] for i in items)
    items = sorted(items, key=lambda i: i["yc"])
    rows = [[items[0]]]
    for it in items[1:]:
        row_y = statistics.mean(i["yc"] for i in rows[-1])
        if abs(it["yc"] - row_y) < 0.6 * med_h:
            rows[-1].append(it)
        else:
            rows.append([it])
    return [sorted(r, key=lambda i: i["x1"]) for r in rows]


def drop_line_numbers(rows):
    """Remove an editor's line-number gutter if present (pure-digit boxes at the far left)."""
    nums = [r[0] for r in rows if len(r) > 1 and re.fullmatch(r"\d{1,4}", r[0]["text"].strip())]
    if len(rows) >= 3 and len(nums) >= 0.5 * len(rows):
        rows = [r[1:] if (len(r) > 1 and re.fullmatch(r"\d{1,4}", r[0]["text"].strip())) else r for r in rows]
    # numbers glued to the text: "12 def foo():"
    glued = sum(1 for r in rows if r and re.match(r"^\d{1,4}\s+\S", r[0]["text"]))
    if len(rows) >= 3 and glued >= 0.6 * len(rows):
        for r in rows:
            if r:
                r[0] = dict(r[0], text=re.sub(r"^\d{1,4}\s+", "", r[0]["text"]))
    return [r for r in rows if r]


def snap_indent(value, all_values):
    """Indentation is measured in pixels, so it is noisy. Snap to the tutorial's unit (4 or 2)."""
    for unit in (4, 2):
        ok = sum(1 for v in all_values if abs(v - unit * round(v / unit)) <= 0.6)
        if ok >= 0.9 * len(all_values):
            return unit * round(value / unit)
    return round(value)


def indent_from_px(x, base, cw):
    return int(round(max(0.0, (x - base) / cw)))


def build_snapshots(frames_items, strip_numbers=True):
    """frames_items: list of (time, items). Returns list of (time, [(line, conf), ...])."""
    all_rows = []
    for t, items in frames_items:
        rows = group_lines(items)
        if strip_numbers:
            rows = drop_line_numbers(rows)
        all_rows.append((t, rows))

    # Character width: in monospace code, two words separated by ONE space start (len+1) chars apart.
    gaps = [(b2["x1"] - b1["x1"]) / (len(b1["text"]) + 1)
            for _, rows in all_rows for r in rows for b1, b2 in zip(r, r[1:]) if len(b1["text"]) >= 2]
    widths = [(b["x2"] - b["x1"]) / max(len(b["text"]), 1)
              for _, rows in all_rows for r in rows for b in r if len(b["text"]) >= 6]
    cw = statistics.median(gaps) if len(gaps) >= 5 else (statistics.median(widths) * 1.08 if widths else 8.0)
    lefts = [min(b["x1"] for r in rows for b in r) for _, rows in all_rows if rows]
    base = statistics.median(lefts) if lefts else 0.0

    raw_indents = [max(0.0, (r[0]["x1"] - base) / cw) for _, rows in all_rows for r in rows]
    snapshots = []
    for t, rows in all_rows:
        lines = []
        for r in rows:
            indent = snap_indent(max(0.0, (r[0]["x1"] - base) / cw), raw_indents) if raw_indents else 0
            text = " " * indent + r[0]["text"]
            for cur in r[1:]:
                col = indent_from_px(cur["x1"], base, cw)  # code is monospace: x position = column
                text += " " * max(1, col - len(text)) + cur["text"]
            lines.append((text, min(b["conf"] for b in r)))
        snapshots.append((t, lines))
    return snapshots


FIXUPS = [  # (pattern, replacement) - small, editable list of typical OCR slips in Python code. Order matters.
    (r"(?<=\S)  +(?=\S)", " "),       # stray double spaces inside a line
    (r"^(\s*)(Class|Def|Import|From|Return|Print|If|Else|Elif|For|While|Try|Except|With)\b",
     lambda m: m.group(1) + m.group(2).lower()),
    (r"\bdef\s+_*\s*(init|str|repr|len|call|enter|exit)_*\s*\(", lambda m: f"def __{m.group(1)}__("),
    (r"\b_*name_*\s*==\s*[\"']_*main_*[\"']", '__name__ == "__main__"'),
    (r"\b(print|len|range|int|str|float|input|open|sum|max|min|sorted)\s+\(", lambda m: m.group(1) + "("),
    (r"\b(def\s+\w+|class\s+\w+)\s+\(", lambda m: m.group(1) + "("),
    (r"\s*\|$", ""),                    # the blinking text cursor read as a pipe
    (r"\s+(?=[)\]},:;])", ""),        # "print(area(10) )" -> "print(area(10))"
]


def apply_fixups(line):
    for pat, rep in FIXUPS:
        line = re.sub(pat, rep, line)
    return line


# ----------------------------------------------- step 6: merge snapshots
def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


class Document:
    """Keeps the best-known version of every line of code seen so far."""

    def __init__(self, fuzzy=88):
        self.lines = []  # each: {"key": str, "obs": [(text, conf)]}
        self.fuzzy = fuzzy

    def _key_for(self, text):
        n = _norm(text)
        for ln in self.lines:
            if fuzz.ratio(n, ln["key"]) >= self.fuzzy:
                return ln["key"]
        return n

    def add_snapshot(self, snap):
        if not snap:
            return
        keys = [self._key_for(t) for t, _ in snap]
        doc_keys = [ln["key"] for ln in self.lines]
        ops = SequenceMatcher(None, doc_keys, keys, autojunk=False).get_opcodes()
        equal_ops = [o for o in ops if o[0] == "equal"]

        if not self.lines:
            self.lines = [{"key": k, "obs": [s]} for k, s in zip(keys, snap)]
            return
        if not equal_ops:  # nothing in common: a new file / new section
            self.lines += [{"key": "---", "obs": [("", 1.0)]}]
            self.lines += [{"key": k, "obs": [s]} for k, s in zip(keys, snap)]
            return

        first_i, last_i = equal_ops[0][1], equal_ops[-1][2]
        new_doc = []
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                for di, sj in zip(range(i1, i2), range(j1, j2)):
                    ln = self.lines[di]
                    ln["obs"].append(snap[sj])
                    new_doc.append(ln)
            elif tag == "insert":
                new_doc += [{"key": keys[j], "obs": [snap[j]]} for j in range(j1, j2)]
            elif tag == "replace":
                new_doc += [{"key": keys[j], "obs": [snap[j]]} for j in range(j1, j2)]
            elif tag == "delete":
                inside = i1 >= first_i and i2 <= last_i
                if not inside:  # outside the visible overlap: probably scrolled out of view
                    new_doc += self.lines[i1:i2]
                # inside the visible overlap: the instructor deleted it
        self.lines = new_doc

    def render(self, low_conf=0.85):
        out, review = [], []
        for n, ln in enumerate(self.lines, 1):
            recent = [t for t, _ in ln["obs"][-3:]]
            text = Counter(recent).most_common(1)[0][0] if len(set(recent)) < len(recent) else recent[-1]
            conf = min(c for t, c in ln["obs"][-3:] if t == text)
            text = apply_fixups(text)
            out.append(text)
            if ln["key"] != "---" and conf < low_conf:
                review.append((len(out), text.strip(), round(conf, 2)))
        return "\n".join(out), review


# ------------------------------------------------------------ main entry
def extract_code_from_frames(frames, strip_numbers=True, engine="tesseract", progress=print):
    """
    Run OCR and document merging on an iterable of (time_seconds, frame) pairs.
    Returns (code_text, review_lines, stats).
    """
    frames_items = []
    for t, frame in frames:
        frames_items.append((t, ocr_frame(frame, engine)))
        progress(f"  stable moment at {t:6.1f}s  ({len(frames_items)} read so far)")
    snaps = build_snapshots(frames_items, strip_numbers)
    doc = Document()
    for _, snap in snaps:
        doc.add_snapshot(snap)
    code, review = doc.render()
    return code, review, {"stable_moments": len(frames_items)}


def extract_code(video, start, end, region=None, samples_per_sec=2.0, min_stable=2,
                 strip_numbers=True, engine="tesseract", progress=print):
    def gen_frames():
        nonlocal last
        for (t, frame), stats in stable_frames(video, start, end, region, samples_per_sec, min_stable):
            last = stats
            yield (t, frame)

    last = {}
    code, review, info = extract_code_from_frames(
        gen_frames(), strip_numbers=strip_numbers, engine=engine, progress=progress
    )
    return code, review, {"sampled": last.get("sampled", 0), "stable_moments": info.get("stable_moments", 0)}

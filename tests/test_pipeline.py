"""
test_pipeline.py
================
Runs extract_code() on synthetic test videos.

1. Strict regression variants: A, B, C must match ground truth 100% exactly (all non-blank lines).
2. Stress variant: D reports weak spots (18px font, short vars, digits 0 & 8) with line-by-line diff,
   without blocking the overall PASS/FAIL.
3. Baseline support: saves or verifies against tests/baseline/ (code text and stable moment counts).

Usage:
    .venv\\Scripts\\python tests\\test_pipeline.py
    .venv\\Scripts\\python tests\\test_pipeline.py --save-baseline
    .venv\\Scripts\\python tests\\test_pipeline.py --compare-baseline
"""

import os
import sys
import json
import difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from videocode_core import extract_code

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baseline")

REGRESSION_VARIANTS = [
    ("variant_a", "variant_a.mp4", "ground_truth_a.txt"),
    ("variant_b", "variant_b.mp4", "ground_truth_b.txt"),
    ("variant_c", "variant_c.mp4", "ground_truth_c.txt"),
]

STRESS_VARIANTS = [
    ("variant_d", "variant_d.mp4", "ground_truth_d.txt"),
]


def non_blank(text):
    """Return list of non-blank lines with leading whitespace/indentation preserved."""
    return [l.rstrip() for l in text.splitlines() if l.strip()]


def compare_strict(got_text, expected_text, label):
    got_lines = non_blank(got_text)
    exp_lines = non_blank(expected_text)
    total = len(exp_lines)
    exact = sum(1 for g, e in zip(got_lines, exp_lines) if g == e)

    print(f"\n{'='*60}")
    print(f"  {label} (STRICT) — {exact}/{total} lines exact")
    print(f"{'='*60}")

    max_lines = max(len(exp_lines), len(got_lines))
    for i in range(max_lines):
        e = exp_lines[i] if i < len(exp_lines) else "<NO LINE>"
        g = got_lines[i] if i < len(got_lines) else "<NO LINE>"
        mark = "✓" if e == g else "✗"
        print(f"  {mark} line {i+1:2d}: {repr(g)}")
        if e != g:
            print(f"       expected: {repr(e)}")

    passed = (len(got_lines) == len(exp_lines)) and (exact == total)
    print(f"\n  → {'PASS' if passed else 'FAIL'}")
    return passed


def compare_stress(got_text, expected_text, label):
    got_lines = non_blank(got_text)
    exp_lines = non_blank(expected_text)
    total = len(exp_lines)
    exact = sum(1 for g, e in zip(got_lines, exp_lines) if g == e)

    print(f"\n{'='*60}")
    print(f"  {label} (STRESS REPORT - Non-blocking) — {exact}/{total} lines exact")
    print(f"{'='*60}")

    max_lines = max(len(exp_lines), len(got_lines))
    for i in range(max_lines):
        e = exp_lines[i] if i < len(exp_lines) else "<NO LINE>"
        g = got_lines[i] if i < len(got_lines) else "<NO LINE>"
        mark = "✓" if e == g else "✗"
        print(f"  {mark} line {i+1:2d}: {repr(g)}")
        if e != g:
            print(f"       expected: {repr(e)}")
            diff = list(difflib.ndiff([e], [g]))
            print(f"       diff: {' | '.join(d.strip() for d in diff)}")

    print(f"\n  → REPORTED (Stress test does not block test suite)")
    return True


def save_baseline(results):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    for label, code, stats in results:
        code_file = os.path.join(BASELINE_DIR, f"{label}_code.txt")
        stats_file = os.path.join(BASELINE_DIR, f"{label}_stats.json")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  [BASELINE SAVED] {label}: {code_file}, {stats_file}")


def check_baseline(results):
    all_match = True
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPARISON")
    print(f"{'='*60}")
    for label, code, stats in results:
        code_file = os.path.join(BASELINE_DIR, f"{label}_code.txt")
        stats_file = os.path.join(BASELINE_DIR, f"{label}_stats.json")
        if not os.path.exists(code_file) or not os.path.exists(stats_file):
            print(f"  ✗ {label}: baseline files missing")
            all_match = False
            continue
        with open(code_file, "r", encoding="utf-8") as f:
            base_code = f.read()
        with open(stats_file, "r", encoding="utf-8") as f:
            base_stats = json.load(f)

        code_match = (code == base_code)
        moments_match = (stats.get("stable_moments") == base_stats.get("stable_moments"))
        if code_match and moments_match:
            print(f"  ✓ {label}: code identical, stable moments match ({stats['stable_moments']})")
        else:
            all_match = False
            print(f"  ✗ {label}: MISMATCH with baseline!")
            if not code_match:
                print(f"     Code diff: {len(code)} chars vs {len(base_code)} baseline chars")
            if not moments_match:
                print(f"     Moments diff: got {stats.get('stable_moments')}, baseline had {base_stats.get('stable_moments')}")
    return all_match


def main():
    save_base_flag = "--save-baseline" in sys.argv
    check_base_flag = "--compare-baseline" in sys.argv

    all_passed = True
    baseline_results = []

    # 1. Strict Regression Variants (A, B, C)
    for label, mp4, gt_file in REGRESSION_VARIANTS:
        video_path = os.path.join(FIXTURES, mp4)
        gt_path = os.path.join(FIXTURES, gt_file)

        if not os.path.exists(video_path):
            print(f"\n[SKIP] {label}: {video_path} not found")
            all_passed = False
            continue

        with open(gt_path, encoding="utf-8") as f:
            expected = f.read()

        print(f"\nRunning extract_code on {label} ...")
        code, review, stats = extract_code(
            video_path,
            start=0,
            end=99999,
            region=None,
            samples_per_sec=2.0,
            min_stable=2,
            progress=lambda msg: print(f"  {msg}"),
        )
        print(f"  stats: {stats}")
        baseline_results.append((label, code, stats))

        ok = compare_strict(code, expected, label)
        if not ok:
            all_passed = False

    # 2. Stress Variant (D)
    for label, mp4, gt_file in STRESS_VARIANTS:
        video_path = os.path.join(FIXTURES, mp4)
        gt_path = os.path.join(FIXTURES, gt_file)

        if not os.path.exists(video_path):
            print(f"\n[SKIP] {label}: {video_path} not found")
            continue

        with open(gt_path, encoding="utf-8") as f:
            expected = f.read()

        print(f"\nRunning extract_code on {label} (STRESS) ...")
        code, review, stats = extract_code(
            video_path,
            start=0,
            end=99999,
            region=None,
            samples_per_sec=2.0,
            min_stable=2,
            progress=lambda msg: print(f"  {msg}"),
        )
        print(f"  stats: {stats}")
        compare_stress(code, expected, label)

    # 3. Handle baseline
    if save_base_flag:
        save_baseline(baseline_results)
    elif check_base_flag or os.path.exists(BASELINE_DIR):
        if not check_baseline(baseline_results):
            all_passed = False

    print("\n" + "="*60)
    print(f"  OVERALL: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    print("="*60 + "\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

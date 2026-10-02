"""
Part 3a — Captions (OCR)
========================
Reads the text burned into the video (TikTok/Instagram captions, repost
headers) and turns it into timestamped segments:

    {"start_s": 0.0, "end_s": 2.5, "text": "Help him out bro.", "conf": 82}

Why captions matter: many clips have music instead of speech, and the
caption is the poster's narration of what's happening. Treat it as
CONTEXT (someone's story about the clip), not as evidence of what the
robot said or did. Part 3b (Whisper) provides the audio evidence.

How it works:
  1. Sample frames every SAMPLE_EVERY_S seconds inside the content range
     (outros trimmed by Part 2).
  2. Enlarge each frame 3x (Tesseract struggles with small text), then make
     two black-on-white versions: one keeping near-WHITE pixels (the usual
     white caption text) and one keeping near-BLACK pixels (black text on
     a white caption box). Tesseract reads best as dark text on light.
  3. Run Tesseract, which returns every word it thinks it sees plus a
     confidence score 0-100. Most junk (textures, logos, sidewalk cracks
     read as letters) scores under ~55; real caption words score 70-96.
  4. Keep a line only if its average confidence >= MIN_LINE_CONF and it has
     at least 2 real words. Drop @handles (watermarks).
  5. Merge: the same caption shows on many consecutive frames, so fuzzy-
     match text across frames and collapse repeats into one segment with a
     start and end time.

Outputs:
  data/captions/<id>.json    caption segments for one video
  data/captions.csv          all videos, one row per segment

Usage:
  python captions.py            # videos not yet processed
  python captions.py --force
Requires the `tesseract` command (Mac: brew install tesseract).
"""

import argparse
import csv
import io
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).parent
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_DIR = ROOT / "data" / "captions"
ALL_CSV = ROOT / "data" / "captions.csv"

SAMPLE_EVERY_S = 0.5    # OCR two frames per second
UPSCALE = 3             # enlarge frames before OCR
WHITE_LEVEL = 215       # gray >= this counts as "white text"
BLACK_LEVEL = 40        # gray <= this counts as "black text"
MIN_WORD_CONF = 30      # words below this are dropped before line scoring
MIN_LINE_CONF = 70      # a line needs this average confidence to be kept
SAME_TEXT = 0.75        # similarity above this = same caption (see similar())
MAX_GAP_S = 1.0         # a caption missing for <= this long is still one segment

WORD_RE = re.compile(r"[A-Za-z]{2,}")


def ocr_lines(img) -> list[tuple[str, float]]:
    """Run Tesseract on one black-on-white image; return (line_text, mean_conf)."""
    ok, png = cv2.imencode(".png", img)
    out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11", "tsv"],
                         input=png.tobytes(), capture_output=True, check=True)
    rows = csv.DictReader(io.StringIO(out.stdout.decode("utf-8", "ignore")), delimiter="\t")
    lines: dict[tuple, list] = {}
    for r in rows:
        text = (r.get("text") or "").strip()
        try:
            conf = float(r["conf"])
        except (KeyError, ValueError, TypeError):
            continue
        if not text or conf < MIN_WORD_CONF:
            continue
        key = (r["block_num"], r["par_num"], r["line_num"])
        lines.setdefault(key, []).append((int(r["left"]), text, conf))

    result = []
    for words in lines.values():
        words.sort()                                    # left to right
        text = " ".join(w for _, w, _ in words)
        conf = sum(c for *_, c in words) / len(words)
        real_words = WORD_RE.findall(text)
        if conf >= MIN_LINE_CONF and len(real_words) >= 2 and "@" not in text:
            result.append((text, conf))
    return result


def ocr_frame(frame) -> tuple[str, float]:
    """OCR one video frame in both polarities; return (text, conf) or ("", 0)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    # white text -> black on white:  pixels >= WHITE_LEVEL become black
    _, white_text = cv2.threshold(gray, WHITE_LEVEL, 255, cv2.THRESH_BINARY_INV)
    # black text -> black on white:  pixels <= BLACK_LEVEL stay black
    _, black_text = cv2.threshold(gray, BLACK_LEVEL, 255, cv2.THRESH_BINARY)

    best = ("", 0.0)
    for img in (white_text, black_text):
        if best[0] and best[1] >= MIN_LINE_CONF + 10:
            break                             # white text found cleanly: skip the 2nd pass (2x faster)
        lines = ocr_lines(img)
        if not lines:
            continue
        # Top-to-bottom order is kept by Tesseract's block order
        text = " / ".join(t for t, _ in lines)
        conf = sum(c for _, c in lines) / len(lines)
        # prefer whichever polarity found more confident text
        if len(text) * conf > len(best[0]) * best[1]:
            best = (text, conf)
    return best


def similar(a: str, b: str) -> float:
    """How alike two OCR readings are, 0-1.

    OCR of the SAME caption varies frame to frame ("delivery rob" vs
    "delivery robot", a stray extra word), so plain character similarity
    splits one caption into several. We also compute word overlap: the
    share of the shorter reading's words that appear in the longer one.
    """
    char_sim = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    wa, wb = set(WORD_RE.findall(a.lower())), set(WORD_RE.findall(b.lower()))
    overlap = len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0
    return max(char_sim, overlap)


def merge(samples: list[tuple[float, str, float]], step: float) -> list[dict]:
    """Collapse runs of the same caption into segments."""
    segs: list[dict] = []
    for t, text, conf in samples:
        if not text:
            continue
        last = segs[-1] if segs else None
        if last and t - last["end_s"] <= MAX_GAP_S + 1e-6 and similar(text, last["_best"]) >= SAME_TEXT:
            last["end_s"] = round(t + step, 2)
            last["n_frames"] += 1
            # keep the most complete confident reading: OCR sometimes drops
            # words at the edges, so favour (real words x confidence)
            if len(WORD_RE.findall(text)) * conf > len(WORD_RE.findall(last["_best"])) * last["conf"]:
                last["_best"], last["conf"] = text, conf
        else:
            segs.append({"start_s": round(t, 2), "end_s": round(t + step, 2),
                         "_best": text, "conf": conf, "n_frames": 1})
    out = []
    for s in segs:
        out.append({"start_s": s["start_s"], "end_s": s["end_s"], "text": s["_best"],
                    "conf": round(s["conf"], 1), "n_frames": s["n_frames"]})
    return out


def process(video: Path, start: float, end: float) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    samples = []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        text, conf = ocr_frame(frame)
        samples.append((round(t, 2), text, conf))
        t += SAMPLE_EVERY_S
    cap.release()
    segs = merge(samples, SAMPLE_EVERY_S)
    # a caption seen on only one frame is usually noise
    return [s for s in segs if s["n_frames"] >= 2]


def main(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    rows = []
    for _, m in manifest.iterrows():
        out = OUT_DIR / f"{m.video_id}.json"
        if out.exists() and not force:
            print(f"skip  {m.video_id}")
            segs = json.loads(out.read_text())
        else:
            start = m.content_start_s if pd.notna(m.get("content_start_s")) else 0.0
            end = m.content_end_s if pd.notna(m.get("content_end_s")) else m.duration_s
            print(f"ocr   {m.video_id} ({m.original_file}) {start:.1f}-{end:.1f}s")
            segs = process(ROOT / m.normalized_path, float(start), float(end))
            out.write_text(json.dumps(segs, indent=2))
            for s in segs:
                print(f"   {s['start_s']:6.1f}-{s['end_s']:6.1f}s  conf={s['conf']:4.0f}  {s['text']}")
            if not segs:
                print("   (no captions found)")
        rows += [{"video_id": m.video_id, **s} for s in segs]
    pd.DataFrame(rows).to_csv(ALL_CSV, index=False)
    print(f"\nWrote {ALL_CSV} ({len(rows)} caption segments)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)

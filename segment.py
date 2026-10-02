"""
Part 2 — Segment
================
Finds the camera cuts in each video and trims non-content (outro/intro
cards, frozen frames) so later steps only see real footage.

IMPORTANT: shots are NOT episodes. A shot is "the camera didn't cut".
An episode is "one social moment". One shot can hold several episodes,
and one episode can span several shots. Shots are just hints that
Part 4 (labeling) uses.

Step 0 — find the footage box: reposted clips are often a small video
window inside a black frame with a caption header. All measurements below
are taken inside that window only (see footage_box()).

How it works — three numbers per frame:
  hist_diff  How different the COLOR MIX is from the previous frame
             (0 = identical, 1 = totally different). A hard cut to a new
             scene makes this jump. Camera motion barely moves it,
             because the same things are still in view.
  motion     Average pixel change vs the previous frame (0-255 scale).
             High = things moving or camera moving. ~0 = frozen image.
  brightness Average gray level. Outro cards are usually dark overlays.

Rules (tuned on real Instagram/TikTok clips — adjust if they misfire):
  cut     hist_diff > CUT_THRESHOLD
  static  motion < STATIC_MOTION for at least STATIC_MIN_S seconds
  outro   static AND at the end AND (dark OR cut off from the footage by a
          hard cut) -> trimmed.
  intro   static AND at the start AND dark -> trimmed (stricter on purpose).
  Other static shots are kept and flagged "static" (e.g., a tripod shot of
  a robot waiting at a door, which is exactly footage we want).

Outputs:
  data/shots.csv              one row per shot
  data/signals/<id>.csv       per-frame numbers, for debugging thresholds
  data/contact/<id>.jpg       1 frame/sec thumbnails with timestamps (for you)
  videos/content/<id>.mp4     the video with intro/outro removed (only written
                              when something was trimmed; manifest.content_path
                              always points at the clip to use)
  manifest.csv                gains content_start_s, content_end_s, n_shots,
                              footage_box (where the real video sits in the frame)

Usage:
  python segment.py             # every video in the manifest not yet segmented
  python segment.py --force     # redo all
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MANIFEST = ROOT / "data" / "manifest.csv"
SHOTS = ROOT / "data" / "shots.csv"
SIGNAL_DIR = ROOT / "data" / "signals"
CONTACT_DIR = ROOT / "data" / "contact"
CONTENT_DIR = ROOT / "videos" / "content"

CUT_THRESHOLD = 0.35   # hist_diff above this = hard cut
STATIC_MOTION = 1.0    # motion below this = frame is frozen
STATIC_MIN_S = 1.0     # frozen this long = static segment
MIN_SHOT_S = 0.5       # ignore "shots" shorter than this (flashes, fades)
DARK_BRIGHTNESS = 60   # mean gray level below this = dark overlay (end cards)
FADE_MAX_S = 1.0       # a dark sliver shorter than this before an outro is its fade-in
BOX_PIXEL_STD = 12     # a pixel "varies" if its gray level std over time > this
BOX_LINE_FRAC = 0.05   # a row/column is footage if > 5% of its pixels vary


def footage_box(video: Path, samples: int = 40) -> tuple[int, int, int, int]:
    """Find where the real footage sits inside the frame.

    Reposted clips are often "letterboxed": a small video window inside a big
    black frame, with a caption header above it. Measured over the whole
    frame, that black border drags brightness down and dilutes motion, so a
    quiet shot looks like a dark, frozen end card. Fix: sample frames across
    the video and keep only the rows/columns whose pixels actually CHANGE over
    time. Static borders and static caption text fall outside the box.
    Returns (x, y, w, h); the full frame if nothing clear is found.
    """
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    H, W = frames[0].shape
    varies = np.stack(frames).std(axis=0) > BOX_PIXEL_STD
    rows = np.where(varies.mean(axis=1) > BOX_LINE_FRAC)[0]
    cols = np.where(varies.mean(axis=0) > BOX_LINE_FRAC)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, W, H
    x, y, w, h = cols.min(), rows.min(), cols.max() - cols.min() + 1, rows.max() - rows.min() + 1
    if w * h < 0.2 * W * H:                # suspiciously small: don't trust it
        return 0, 0, W, H
    return int(x), int(y), int(w), int(h)


def frame_signals(video: Path, box: tuple[int, int, int, int]) -> pd.DataFrame:
    """Read every frame and compute hist_diff, motion, brightness INSIDE box."""
    bx, by, bw, bh = box
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, prev_hist, prev_small, i = [], None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[by:by + bh, bx:bx + bw]
        # Color histogram over hue x saturation (ignores brightness shifts)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        # Tiny grayscale copy: fast, and ignores compression noise
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 112)).astype(np.float32)

        if prev_hist is None:
            hist_diff, motion = 0.0, 0.0
        else:
            hist_diff = 1 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            motion = float(np.mean(np.abs(small - prev_small)))
        rows.append({"frame": i, "t": i / fps, "hist_diff": max(0.0, hist_diff),
                     "motion": motion, "brightness": float(small.mean())})
        prev_hist, prev_small, i = hist, small, i + 1
    cap.release()
    return pd.DataFrame(rows), fps


def find_shots(sig: pd.DataFrame, fps: float, duration: float) -> list[dict]:
    """Split at cuts, then split out long frozen stretches."""
    # 1) Hard cuts
    cut_times = sig.loc[sig["hist_diff"] > CUT_THRESHOLD, "t"].tolist()
    bounds = [0.0]
    for t in cut_times:
        if t - bounds[-1] >= MIN_SHOT_S:   # ignore cuts right after a cut (flashes)
            bounds.append(t)
    bounds.append(duration)

    # 2) Frozen stretches, found as runs of low-motion frames
    frozen = (sig["motion"] < STATIC_MOTION).to_numpy().copy()
    frozen[0] = False                      # frame 0 has no "previous", so motion=0 means nothing
    runs, start = [], None
    for i, f in enumerate(frozen):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(frozen)))
    runs = [(0 if a == 1 else a, b) for a, b in runs]   # a run from frame 1 really starts at 0
    static_spans = [(sig.t[a], sig.t[b - 1] + 1 / fps) for a, b in runs
                    if (b - a) / fps >= STATIC_MIN_S]
    for a, b in static_spans:              # make static spans their own shots
        bounds += [a, min(b, duration)]
    bounds = sorted(set(round(b, 3) for b in bounds))

    # 3) Describe each shot
    shots = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 1 / fps:
            continue
        s = sig[(sig.t >= a) & (sig.t < b)]
        # skip the first frame's hist_diff: it IS the cut into this shot
        is_static = any(sa <= a + 1e-3 and b <= sb + 1e-3 for sa, sb in static_spans)
        shots.append({
            "start_s": round(a, 2), "end_s": round(b, 2), "duration_s": round(b - a, 2),
            "mean_motion": round(s.motion.iloc[1:].mean() if len(s) > 1 else 0, 2),
            "max_motion": round(s.motion.iloc[1:].max() if len(s) > 1 else 0, 2),
            "mean_brightness": round(s.brightness.mean(), 1),
            "static": is_static,
        })

    # 4) Label each shot.
    # Being frozen is NOT enough to call something an outro: a tripod shot of a
    # robot silently waiting at a door is also frozen, and that's exactly the
    # footage we want to keep. So an intro/outro must be frozen AND either dark
    # (platform end cards are dim overlays) or separated from the footage by a
    # hard cut. Short dark slivers (< FADE_MAX_S) before an outro are its fade-in.
    cuts = set(round(t, 2) for t in bounds[1:-1] if t in [round(c, 3) for c in cut_times])
    for s in shots:
        s["kind"] = "static" if s["static"] else "content"

    def is_card(s, edge_time):
        return s["static"] and (s["mean_brightness"] < DARK_BRIGHTNESS or edge_time in cuts)

    # Intros get the stricter test (dark only): cutting real footage is a worse
    # mistake than keeping a title card, which the VLM can simply ignore.
    for k in range(len(shots)):                       # intro: walk forward from the start
        s = shots[k]
        if s["static"] and s["mean_brightness"] < DARK_BRIGHTNESS:
            s["kind"] = "intro"
        else:
            break
    for k in range(len(shots) - 1, -1, -1):          # outro: walk back from the end
        s = shots[k]
        if is_card(s, s["start_s"]) or (s["duration_s"] < FADE_MAX_S and s["mean_brightness"] < DARK_BRIGHTNESS):
            s["kind"] = "outro"
        else:
            break
    for s in shots:
        del s["static"]
    return shots


def contact_sheet(video: Path, out: Path, every_s: float = 1.0, cols: int = 7) -> None:
    """1 thumbnail per second with its timestamp burned in, for quick review."""
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thumbs = []
    for f in range(0, n, max(1, int(round(fps * every_s)))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (180, int(180 * img.shape[0] / img.shape[1])))
        label = f"{f / fps:.0f}s"
        cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        thumbs.append(img)
    cap.release()
    if not thumbs:
        return
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    grid = np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)])
    cv2.imwrite(str(out), grid)


def trim(video: Path, out: Path, start: float, end: float) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start}", "-to", f"{end}",
                    "-i", str(video), "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", str(out)], check=True)


def main(force: bool = False) -> None:
    for d in (SIGNAL_DIR, CONTACT_DIR, CONTENT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    for col in ("content_start_s", "content_end_s", "n_shots", "footage_box", "content_path"):
        if col not in manifest:
            manifest[col] = np.nan
    all_shots = pd.read_csv(SHOTS) if SHOTS.exists() and not force else pd.DataFrame()

    for i, row in manifest.iterrows():
        vid = row.video_id
        if not force and pd.notna(row.n_shots):
            print(f"skip  {vid} (already segmented)")
            continue
        video = ROOT / row.normalized_path
        print(f"segment {vid} ({row.original_file})")

        box = footage_box(video)
        sig, fps = frame_signals(video, box)
        sig.round(4).to_csv(SIGNAL_DIR / f"{vid}.csv", index=False)
        duration = len(sig) / fps
        shots = find_shots(sig, fps, duration)

        content = [s for s in shots if s["kind"] not in ("intro", "outro")]
        c_start = content[0]["start_s"] if content else 0.0
        c_end = content[-1]["end_s"] if content else duration
        # Only write a trimmed copy if something was actually trimmed. When the
        # whole video is content (e.g. TikTok downloads have no end cards), a
        # "trimmed" copy would just duplicate the normalized video and double
        # the disk space; point at the normalized file instead.
        trimmed = c_start > 0.05 or c_end < duration - 0.05
        if trimmed:
            content_file = CONTENT_DIR / f"{vid}.mp4"
            trim(video, content_file, c_start, c_end)
            content_path = str(content_file.relative_to(ROOT))
        else:
            content_path = row.normalized_path
        contact_sheet(video, CONTACT_DIR / f"{vid}.jpg")

        for n, s in enumerate(shots):
            print(f"   shot {n}: {s['start_s']:6.2f}-{s['end_s']:6.2f}s  {s['kind']:<8}"
                  f" motion={s['mean_motion']:5.1f}  bright={s['mean_brightness']:5.1f}")
        print(f"   footage box (x,y,w,h): {box}")
        print(f"   content kept: {c_start:.2f}-{c_end:.2f}s of {duration:.2f}s"
              + ("" if trimmed else "  (nothing trimmed: no copy made)"))

        manifest["footage_box"] = manifest["footage_box"].astype(object)
        manifest["content_path"] = manifest["content_path"].astype(object)
        manifest.at[i, "content_path"] = content_path
        manifest.loc[i, ["content_start_s", "content_end_s", "n_shots"]] = [c_start, c_end, len(shots)]
        manifest.at[i, "footage_box"] = "x={} y={} w={} h={}".format(*box)
        new = pd.DataFrame([{"video_id": vid, "shot": n, **s} for n, s in enumerate(shots)])
        if not all_shots.empty:
            all_shots = all_shots[all_shots.video_id != vid]
        all_shots = pd.concat([all_shots, new], ignore_index=True)
        manifest.to_csv(MANIFEST, index=False)       # save after every video
        all_shots.to_csv(SHOTS, index=False)

    manifest.to_csv(MANIFEST, index=False)
    all_shots.to_csv(SHOTS, index=False)
    print(f"\nWrote {SHOTS} ({len(all_shots)} shots)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)

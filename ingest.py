"""
Part 1 — Ingest
===============
Turns a folder of messy downloaded videos into a clean, uniform set,
plus a manifest (one row per video) that every later step reads.

Why this step exists:
  TikTok/Instagram files come in different frame rates (often variable),
  resolutions, and codecs. Every later tool (scene detection, Whisper,
  frame sampling for the VLM) behaves more predictably on uniform input.

What it does for each video in videos/raw/:
  1. Gives it a stable ID from a hash of its bytes, so re-running the
     script never creates duplicates, even if you rename the file.
  2. Probes it with ffprobe (duration, fps, size, has audio).
  3. Re-encodes it to H.264, constant 30 fps, max 720 px tall.
  4. Extracts a 16 kHz mono WAV, the format Whisper expects.
  5. Writes/updates data/manifest.csv, including the source link and post
     caption when the video came from fetch_videos.py.
  6. Deletes the raw download afterwards if it came from fetch_videos.py
     (it can be re-downloaded from its link). Use --keep-raw to keep it.

Usage:
  python ingest.py                 # process everything new in videos/raw
  python ingest.py --force         # redo all videos
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "videos" / "raw"
NORM_DIR = ROOT / "videos" / "normalized"
AUDIO_DIR = ROOT / "videos" / "audio"
MANIFEST = ROOT / "data" / "manifest.csv"

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
TARGET_FPS = 30
MAX_HEIGHT = 720


def video_id(path: Path) -> str:
    """Stable 10-char ID from file contents (not the filename)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def guess_source(name: str) -> str:
    """Best guess from the filename; fix by hand in the manifest if wrong."""
    n = name.lower()
    if "tiktok" in n or n.startswith("tt"):
        return "tiktok"
    if "insta" in n or "ig_" in n or "reel" in n:
        return "instagram"
    if "youtube" in n or n.startswith("yt"):
        return "youtube"
    return "unknown"


def probe(path: Path) -> dict:
    """Ask ffprobe for duration, fps, resolution and whether there's audio."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    num, den = v.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "duration_s": round(float(info["format"]["duration"]), 2),
        "orig_fps": round(fps, 2),
        "orig_width": int(v["width"]),
        "orig_height": int(v["height"]),
        "has_audio": has_audio,
    }


def normalize(src: Path, dst: Path) -> None:
    """Re-encode to constant 30 fps H.264, at most 720 px tall."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
         # -2 keeps the width even (H.264 needs that) and preserves aspect ratio
         "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)',fps={TARGET_FPS}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k",
         str(dst)],
        check=True,
    )


def extract_audio(src: Path, dst: Path) -> None:
    """16 kHz mono WAV: what Whisper is trained on."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )


def load_sources() -> dict:
    """Source link + post caption per downloaded file, from fetch_videos.py."""
    path = ROOT / "data" / "sources.csv"
    if not path.exists():
        return {}
    src = pd.read_csv(path, dtype=str).fillna("")
    return {r["file"]: r for _, r in src.iterrows()}


def main(force: bool = False, limit: int | None = None, keep_raw: bool = False) -> None:
    for d in (NORM_DIR, AUDIO_DIR, MANIFEST.parent):
        d.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()
    done = set(manifest["video_id"]) if not manifest.empty and not force else set()

    sources = load_sources()
    rows = []
    for src in sorted(RAW_DIR.iterdir()):
        if src.suffix.lower() not in VIDEO_EXTS:
            continue
        vid = video_id(src)
        if vid in done:
            print(f"skip  {src.name} (already ingested as {vid})")
            continue

        if limit is not None and len(rows) >= limit:
            break
        print(f"ingest {src.name} -> {vid}", flush=True)
        meta = probe(src)
        norm = NORM_DIR / f"{vid}.mp4"
        normalize(src, norm)

        audio = ""
        if meta["has_audio"]:
            wav = AUDIO_DIR / f"{vid}.wav"
            extract_audio(norm, wav)
            audio = str(wav.relative_to(ROOT))

        known = sources.get(src.name, {})
        rows.append({
            "video_id": vid,
            "original_file": src.name,
            "source": known.get("platform") or guess_source(src.name),
            "source_url": known.get("url", ""),           # where the clip came from
            "post_caption": known.get("caption", ""),     # the poster's own text
            **meta,
            "normalized_path": str(norm.relative_to(ROOT)),
            "audio_path": audio,
            "notes": "",  # free text for you: robot type, setting, anything odd
        })
        # Save after EVERY video, not once at the end: on a 367-video run, a
        # crash or Ctrl-C at video 200 then loses nothing (re-run resumes).
        new = pd.DataFrame(rows[-1:])
        if not manifest.empty:
            manifest = manifest[~manifest["video_id"].isin(new["video_id"])]
        manifest = pd.concat([manifest, new], ignore_index=True)
        manifest.to_csv(MANIFEST, index=False)

        # Downloaded videos can be fetched again from their saved link, so
        # once converted, the large original isn't needed. Files you added by
        # hand (no saved link) are always kept.
        if known.get("url") and not keep_raw:
            try:
                src.unlink()
                print(f"   deleted raw download (re-fetchable from its link)")
            except OSError as e:
                print(f"   could not delete raw file ({e.strerror}); kept it")

    print(f"\n{len(rows)} new video(s). Manifest has {len(manifest)} row(s): {MANIFEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-process all videos")
    ap.add_argument("--limit", type=int, help="process at most N new videos this run")
    ap.add_argument("--keep-raw", action="store_true",
                    help="don't delete downloaded originals after converting them")
    a = ap.parse_args()
    main(a.force, a.limit, a.keep_raw)

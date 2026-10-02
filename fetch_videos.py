"""
Part 0 — Fetch videos from links (yt-dlp)
=========================================
Reads a list of video links (a Google Sheet or a CSV) and downloads each
video into videos/raw/, ready for ingest.py.

Why a script instead of downloading by hand:
  - The SOURCE LINK is recorded for every file (data/sources.csv). You need
    it for reproducibility, and it's how a dataset gets shared later
    (links + labels; the videos themselves can't be redistributed).
  - Already-downloaded videos are skipped (yt-dlp's "download archive"),
    so you can re-run after a failure, or after adding links to the sheet.
  - One broken or deleted link doesn't stop the run; it's logged in
    data/download_failures.csv and the script moves on.
  - It waits a few random seconds between downloads. Hammering a site gets
    you rate-limited or blocked, and it's simply polite.

Input columns (the names in your sheet; extra columns are ignored):
  Video ID   your own ID, e.g. v001   (optional but recommended)
  Link       the video URL            (required)

Usage (in your Mac Terminal, in the project folder, venv active):
  pip install yt-dlp
  python fetch_videos.py --sheet "<google sheet link>" --limit 5     # try 5 first
  python fetch_videos.py --sheet "<google sheet link>"               # everything
  python fetch_videos.py --csv links.csv                             # or a local CSV

Then run ingest.py as usual. It picks up the source link from data/sources.csv.

Before scaling up: check with your advisor that downloading is OK for this
research under the platform's terms and your IRB status.
"""

import argparse
import csv
import io
import random
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "videos" / "raw"
SOURCES = ROOT / "data" / "sources.csv"
FAILURES = ROOT / "data" / "download_failures.csv"
ARCHIVE = ROOT / "data" / "download_archive.txt"
LINKS = ROOT / "data" / "links.csv"

SLEEP_RANGE_S = (3, 8)      # random pause between downloads
SOURCE_FIELDS = ["file", "sheet_id", "platform", "url", "platform_video_id", "uploader",
                 "upload_date", "duration_s", "caption", "downloaded_at"]


# ---------------------------------------------------------------- reading the link list

def sheet_csv_url(url: str) -> str:
    """Turn a normal Google Sheets link into its CSV-export link (same tab)."""
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", url)
    if not m:
        raise SystemExit("That doesn't look like a Google Sheets link.")
    gid = re.search(r"[#&?]gid=(\d+)", url)
    return (f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
            + (f"&gid={gid.group(1)}" if gid else ""))


def read_links(sheet: str | None, csv_path: str | None) -> list[dict]:
    if sheet:
        # works when the sheet is shared as "anyone with the link can view"
        with urllib.request.urlopen(sheet_csv_url(sheet), timeout=30) as r:
            text = r.read().decode("utf-8")
    else:
        text = Path(csv_path).read_text(encoding="utf-8")
    rows = []
    for raw in csv.DictReader(io.StringIO(text)):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        url = row.get("link") or row.get("url")
        if url and url.startswith("http"):
            rows.append({"sheet_id": row.get("video id", ""), "url": url})
    return rows


def platform_of(url: str) -> str:
    u = url.lower()
    for key, name in (("tiktok", "tiktok"), ("instagram", "instagram"),
                      ("youtube", "youtube"), ("youtu.be", "youtube")):
        if key in u:
            return name
    return "other"


PREFIX = {"tiktok": "tt", "instagram": "ig", "youtube": "yt", "other": "web"}


# ---------------------------------------------------------------- bookkeeping files

def append_row(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def already_recorded() -> set[str]:
    if not SOURCES.exists():
        return set()
    with open(SOURCES, encoding="utf-8") as f:
        return {r["url"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------- downloading

class _QuietLogger:
    """yt-dlp prints its own long ERROR lines; we report failures ourselves."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def download(ydl_cls, item: dict, user_agent: str | None = None) -> dict:
    """Download one link. Returns the sources.csv row. Raises on failure."""
    plat = platform_of(item["url"])
    name = f"{PREFIX[plat]}_{item['sheet_id'] or 'x'}_%(id)s"
    opts = {
        "outtmpl": str(RAW_DIR / f"{name}.%(ext)s"),
        # one file that already has video+audio, preferring mp4. This avoids
        # needing ffmpeg to merge separate video and audio streams.
        "format": "best[ext=mp4][vcodec!=none][acodec!=none]/best[vcodec!=none][acodec!=none]/best",
        "download_archive": str(ARCHIVE),   # yt-dlp skips anything listed here
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "logger": _QuietLogger(),
        "noprogress": True,
    }
    if user_agent is not None:
        # Workaround for TikTok's "Unexpected response from webpage request"
        # (yt-dlp issue #17604, Sept 2026): TikTok blocks the browser
        # impersonation yt-dlp uses by default. An empty or plain user agent
        # makes yt-dlp fall back to its own challenge solver.
        opts["http_headers"] = {"User-Agent": user_agent}
    with ydl_cls(opts) as ydl:
        info = ydl.extract_info(item["url"], download=True)
        if info is None:
            raise RuntimeError("skipped by yt-dlp (already in data/download_archive.txt)")
        path = Path(ydl.prepare_filename(info))
    return {
        "file": path.name, "sheet_id": item["sheet_id"], "platform": plat, "url": item["url"],
        "platform_video_id": info.get("id", ""),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "upload_date": info.get("upload_date", ""),
        "duration_s": info.get("duration", ""),
        # the post's own caption: context for the labeler, like burned-in captions
        "caption": (info.get("description") or info.get("title") or "").replace("\n", " ")[:1000],
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main(sheet, csv_path, limit, start, user_agent=None):
    from yt_dlp import YoutubeDL     # imported here so --help works without it

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES.parent.mkdir(parents=True, exist_ok=True)
    all_links = read_links(sheet, csv_path)
    # keep a copy of the full link list, so the dashboard can show progress
    LINKS.parent.mkdir(parents=True, exist_ok=True)
    with open(LINKS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sheet_id", "url"])
        w.writeheader(); w.writerows(all_links)
    links = all_links[start:]
    done = already_recorded()
    todo = [x for x in links if x["url"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(links)} links in the list, {len(links) - len(todo)} skipped "
          f"(already downloaded or past --limit), {len(todo)} to fetch")

    ok = fail = 0
    for i, item in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {item['sheet_id'] or ''} {item['url']}")
        try:
            row = download(YoutubeDL, item, user_agent)
            append_row(SOURCES, SOURCE_FIELDS, row)
            ok += 1
            print(f"   ok -> videos/raw/{row['file']}")
        except Exception as e:                       # deleted video, private, blocked, ...
            fail += 1
            msg = re.sub(r"\s+", " ", ANSI.sub("", str(e)))[:300]
            append_row(FAILURES, ["sheet_id", "url", "error", "at"],
                       {"sheet_id": item["sheet_id"], "url": item["url"], "error": msg,
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            print(f"   FAILED: {msg[:150]}")
        if i < len(todo):
            time.sleep(random.uniform(*SLEEP_RANGE_S))

    print(f"\nDone: {ok} downloaded, {fail} failed"
          + (f" (see {FAILURES.relative_to(ROOT)})" if fail else "")
          + ". Next: python ingest.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sheet", help="Google Sheets link (shared: anyone with link can view)")
    src.add_argument("--csv", help="local CSV with a Link column")
    ap.add_argument("--limit", type=int, help="fetch at most N videos this run")
    ap.add_argument("--start", type=int, default=0, help="skip the first N rows of the list")
    ap.add_argument("--user-agent", help='e.g. --user-agent "" (TikTok workaround, see download())')
    a = ap.parse_args()
    main(a.sheet, a.csv, a.limit, a.start, a.user_agent)

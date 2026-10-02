"""
Run the whole pipeline in batches and keep the dashboard current
================================================================
One command instead of seven. For each batch it runs:

  fetch -> ingest -> segment -> captions -> transcribe -> label -> dashboard

Every step skips videos it has already done, so you can stop (Ctrl-C) and
re-run any time; it picks up where it left off. The dashboard is refreshed
after every step, so with "Auto-reload" ticked in dashboard.html you can
watch the dataset grow.

Usage (Mac Terminal, project folder, venv active):
  python pipeline.py --sheet "<google sheet link>" --batch 10            # one batch of 10
  python pipeline.py --sheet "<google sheet link>" --batch 10 --all      # keep going until done
  python pipeline.py                                                     # no new downloads; process what's here
  python pipeline.py --no-label                                          # everything except the VLM step
  python pipeline.py --backend claude                                    # label with Claude instead of Gemini

Needs: ffmpeg + tesseract (brew install ffmpeg tesseract), and the Python
packages from the README. Missing tools are reported up front, and steps
that need them are skipped rather than crashing the run.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def key_present(key: str) -> bool:
    """True if KEY is set in the environment or has a real value in .env."""
    if os.environ.get(key):
        return True
    env = ROOT / ".env"
    if not env.exists():
        return False
    for line in env.read_text(errors="ignore").splitlines():
        if line.strip().startswith(key + "="):
            val = line.split("=", 1)[1].strip().strip("'\"")
            return bool(val) and "paste-your-key" not in val
    return False


def check_tools(backend: str, labeling: bool) -> dict:
    """What can run on this machine? Returns {step: reason it can't run}."""
    blocked = {}
    if not shutil.which("ffmpeg"):
        for s in ("ingest", "segment"):
            blocked[s] = "ffmpeg not installed (brew install ffmpeg)"
    if not shutil.which("tesseract"):
        blocked["captions"] = "tesseract not installed (brew install tesseract)"
    if not have_module("faster_whisper"):
        blocked["transcribe"] = "faster-whisper not installed (pip install faster-whisper)"
    if not have_module("yt_dlp"):
        blocked["fetch"] = "yt-dlp not installed (pip install yt-dlp)"
    if labeling:
        mod, key = {"gemini": ("google.genai", "GEMINI_API_KEY"),
                    "claude": ("anthropic", "ANTHROPIC_API_KEY")}.get(backend, (None, None))
        if mod and not have_module(mod.split(".")[0]):
            blocked["label"] = f"{mod} not installed"
        elif key and not key_present(key):
            blocked["label"] = f"{key} missing from .env"
    return blocked


def run(step: str, args: list[str], blocked: dict, log: list) -> bool:
    if step in blocked:
        print(f"\n--- {step}: SKIPPED ({blocked[step]})")
        log.append((step, "skipped"))
        return False
    print(f"\n--- {step}: python {' '.join(args)}", flush=True)
    t0 = time.time()
    code = subprocess.call([PY, *args], cwd=ROOT)
    status = "ok" if code == 0 else f"exit {code}"
    print(f"--- {step}: {status} ({time.time() - t0:.0f}s)")
    log.append((step, status))
    subprocess.call([PY, "dashboard.py"], cwd=ROOT, stdout=subprocess.DEVNULL)
    return code == 0


def downloaded_count() -> int:
    src = ROOT / "data" / "sources.csv"
    return max(0, sum(1 for _ in open(src, encoding="utf-8")) - 1) if src.exists() else 0


def one_batch(a, blocked) -> int:
    log = []
    before = downloaded_count()
    if a.sheet or a.csv:
        args = ["fetch_videos.py", "--limit", str(a.batch), "--user-agent", a.user_agent]
        args += ["--sheet", a.sheet] if a.sheet else ["--csv", a.csv]
        run("fetch", args, blocked, log)
    run("ingest", ["ingest.py"], blocked, log)
    run("segment", ["segment.py"], blocked, log)
    run("captions", ["captions.py"], blocked, log)
    run("transcribe", ["transcribe.py"], blocked, log)
    if not a.no_label:
        run("label", ["label.py", "--backend", a.backend], blocked, log)
    subprocess.call([PY, "dashboard.py"], cwd=ROOT)
    print("\nBatch summary: " + ", ".join(f"{s} {st}" for s, st in log))
    return downloaded_count() - before


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--sheet", help="Google Sheets link with a Link column")
    src.add_argument("--csv", help="local CSV with a Link column")
    ap.add_argument("--batch", type=int, default=10, help="videos to download per batch (default 10)")
    ap.add_argument("--all", action="store_true", help="repeat batches until no new videos download")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "claude", "mock"])
    ap.add_argument("--no-label", action="store_true", help="skip the VLM labeling step")
    ap.add_argument("--user-agent", default="", help='passed to fetch_videos.py (default "" = TikTok workaround)')
    a = ap.parse_args()

    blocked = check_tools(a.backend, not a.no_label)
    if blocked:
        print("Heads up, these steps will be skipped:")
        for s, why in blocked.items():
            print(f"  {s:<10} {why}")

    n = 1
    while True:
        print(f"\n========== batch {n} ==========")
        new = one_batch(a, blocked)
        if not a.all or not (a.sheet or a.csv) or new == 0:
            break
        n += 1
    print("\nOpen dashboard.html in your browser (tick Auto-reload to watch it update).")


if __name__ == "__main__":
    main()

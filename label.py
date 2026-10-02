"""
Part 4 — Label (VLM drafts, you review)
=======================================
Sends each trimmed clip plus its captions and transcript to a vision-language
model, and gets back one labeled row per EPISODE (see schema.py).

The VLM PROPOSES labels; you correct them. Its output is a first draft, not
ground truth. Part 5 (compare.py) measures how often it agrees with you.

Backends (pick with --backend):
  gemini   sends the VIDEO FILE itself (with audio). Sees motion between
           frames. Default.
  claude   sends still FRAMES (2 per second) + the text tracks. Can miss
           events shorter than the gap between frames.
  mock     no API call: returns a fixed fake answer. For testing the plumbing.

API keys are read from a file named .env in this folder (never from code):
  GEMINI_API_KEY=...
  ANTHROPIC_API_KEY=...
Don't share or commit .env. (.gitignore already excludes it.)

Usage:
  python label.py --dry-run                 # show exactly what would be sent; no API call
  python label.py                           # label every video not yet labeled (Gemini)
  python label.py --only 9d85a77379         # one video
  python label.py --backend claude          # second opinion, for Part 5
  python label.py --force                   # relabel

Setup (once, in your venv):
  pip install google-genai          # for --backend gemini
  pip install anthropic             # for --backend claude

Outputs:
  data/labels_vlm/<video_id>__<backend>.json   full response + validation problems
  data/labels_vlm.csv                          all episodes, one row each
"""

import argparse
import base64
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from schema import EPISODE_FIELDS, schema_as_text, validate_episode

ROOT = Path(__file__).parent
MANIFEST = ROOT / "data" / "manifest.csv"
PROMPT_FILE = ROOT / "prompts" / "label_prompt.md"
OUT_DIR = ROOT / "data" / "labels_vlm"
ALL_CSV = ROOT / "data" / "labels_vlm.csv"
DRY_DIR = ROOT / "data" / "dry_run"

# Pinned versions, not aliases like "gemini-flash-latest": an alias silently
# switches to a newer model when Google releases one, so labels from March and
# May would come from different models without you knowing. Pin, and change the
# version deliberately (then relabel, and note it).
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "claude": "claude-sonnet-5", "mock": "mock"}
# Frames per second the model sees. Gemini's own default is 1, which can miss a
# half-second bump. Higher = catches faster events but costs more. Override
# with --fps to experiment.
FPS = {"gemini": 2.0, "claude": 2.0, "mock": 2.0}
CLAUDE_MAX_FRAMES = 60  # cap, to bound cost on long clips
INLINE_LIMIT_MB = 18    # bigger videos go through Gemini's upload API instead


# ---------------------------------------------------------------- setup helpers

def load_env() -> None:
    """Read KEY=VALUE lines from .env into the environment (keys already set win)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def prompt_version(text: str) -> str:
    """Short fingerprint of the prompt. Stored with every label, so when you
    edit the prompt later you can tell which labels came from which version."""
    return hashlib.sha1(text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- building the context

def text_tracks(video_id: str, offset: float, length: float) -> tuple[str, str]:
    """Captions and transcript as timestamped lines, shifted to clip time.

    Parts 3a/3b timestamped against the full video; the clip we send starts at
    content_start_s, so every time is shifted by that offset.
    """
    def clip_time(t):
        return round(t - offset, 1)

    cap_lines = []
    cap_file = ROOT / "data" / "captions" / f"{video_id}.json"
    if cap_file.exists():
        for c in json.loads(cap_file.read_text()):
            a, b = clip_time(c["start_s"]), clip_time(c["end_s"])
            if b > 0 and a < length:
                cap_lines.append(f"[{max(a, 0):.1f}-{min(b, length):.1f}s] {c['text']}")

    tr_lines = []
    tr_file = ROOT / "data" / "transcripts" / f"{video_id}.json"
    audio_type = "unknown (no transcript yet)"
    if tr_file.exists():
        tr = json.loads(tr_file.read_text())
        audio_type = tr.get("audio_type", "unknown")
        for s in sorted(tr.get("segments", []), key=lambda s: s["start_s"]):
            a, b = clip_time(s["start_s"]), clip_time(s["end_s"])
            if b <= 0 or a >= length:
                continue
            tag = "reliable" if s["reliable"] else (
                "listen_check" if s.get("flag") == "listen_check" else "unreliable")
            tr_lines.append(f"[{max(a, 0):.1f}-{min(b, length):.1f}s] [{tag}] {s['text']}")

    captions = "\n".join(cap_lines) or "(no captions found)"
    man = pd.read_csv(MANIFEST, dtype=str).fillna("")
    post = man.loc[man.video_id == video_id, "post_caption"] if "post_caption" in man else []
    if len(post) and post.iloc[0].strip():
        captions = (f"Post caption (the text the poster wrote under the video; context, "
                    f"not evidence): {post.iloc[0].strip()}\n\nOn-screen text:\n" + captions)
    transcript = f"Audio type: {audio_type}\n" + ("\n".join(tr_lines) or "(no speech segments)")
    return captions, transcript


def build_prompt(template: str, video_id: str, offset: float, length: float) -> str:
    captions, transcript = text_tracks(video_id, offset, length)
    return template.format(schema=schema_as_text(), captions=captions, transcript=transcript)


def sample_frames(clip: Path, fps: float, max_frames: int) -> list[tuple[float, bytes]]:
    """JPEG frames at `fps`, long side 768 px, each with its timestamp."""
    import cv2
    cap = cv2.VideoCapture(str(clip))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(vfps / fps)))
    idxs = list(range(0, n, step))
    if len(idxs) > max_frames:                      # spread evenly if over the cap
        idxs = [idxs[int(i * len(idxs) / max_frames)] for i in range(max_frames)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        scale = 768 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append((round(i / vfps, 2), jpg.tobytes()))
    cap.release()
    return frames


# ---------------------------------------------------------------- backends

def call_gemini(model: str, clip: Path, prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    size_mb = clip.stat().st_size / 1e6
    if size_mb <= INLINE_LIMIT_MB:
        blob = types.Blob(data=clip.read_bytes(), mime_type="video/mp4")
        try:
            video = types.Part(inline_data=blob, video_metadata=types.VideoMetadata(fps=FPS["gemini"]))
        except (AttributeError, TypeError):          # older SDK without fps control
            video = types.Part(inline_data=blob)
    else:
        f = client.files.upload(file=str(clip))
        while not f.state or f.state.name != "ACTIVE":
            if f.state and f.state.name == "FAILED":
                raise RuntimeError("Gemini could not process the uploaded video")
            time.sleep(3)
            f = client.files.get(name=f.name)
        video = f

    resp = client.models.generate_content(
        model=model,
        contents=[video, prompt],
        config=types.GenerateContentConfig(
            temperature=0,                           # same input -> same labels, as far as possible
            response_mime_type="application/json",
            # we give the model no tools, so switch off "automatic function
            # calling" (silences the SDK's AFC warning; changes nothing else)
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    # the exact model version that answered (matters if you use an alias like gemini-flash-latest)
    u = getattr(resp, "usage_metadata", None)
    usage = {"input_tokens": getattr(u, "prompt_token_count", None),
             "output_tokens": getattr(u, "candidates_token_count", None),
             "thinking_tokens": getattr(u, "thoughts_token_count", None)} if u else {}
    return resp.text, getattr(resp, "model_version", None) or model, usage


def call_claude(model: str, clip: Path, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content = [{"type": "text", "text":
                f"The clip, as frames sampled {FPS['claude']:g} per second. Each frame is "
                f"preceded by its timestamp. You cannot hear the audio; rely on the transcript."}]
    for t, jpg in sample_frames(clip, FPS["claude"], CLAUDE_MAX_FRAMES):
        content.append({"type": "text", "text": f"t={t:.1f}s"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(jpg).decode()}})
    content.append({"type": "text", "text": prompt})
    msg = client.messages.create(model=model, max_tokens=8000, temperature=0,
                                 messages=[{"role": "user", "content": content}])
    u = getattr(msg, "usage", None)
    usage = {"input_tokens": getattr(u, "input_tokens", None),
             "output_tokens": getattr(u, "output_tokens", None)} if u else {}
    return "".join(b.text for b in msg.content if b.type == "text"), getattr(msg, "model", model), usage


def call_mock(model: str, clip: Path, prompt: str) -> str:
    """Deliberately imperfect fake answer, so validation has something to catch."""
    return json.dumps({
        "video_summary": "MOCK: a person helps a delivery robot.",
        "robot_type": "sidewalk delivery robot", "setting": "sidewalk",
        "episodes": [
            {"episode_id": 1, "start_s": 0.0, "end_s": 5.0, "scenario": "s4",
             "scenario_other": "", "trigger": "robot facing the wrong way",
             "robot_actions": "stopped", "robot_speech": "none",
             "human_actions": "person turns the robot around", "outcome": "Fail",
             "failure_types": ["no_help_request", "no_thanks"],
             "should_have": "ask for help, then thank",
             "cf_actions": ["ask_for_help", "thank"], "cf_utterance": "Excuse me, could you turn me around? ... Thank you!",
             "cf_addressee": "nearest_person", "cf_timing": "after_waiting",
             "evidence": [{"t": 1.0, "source": "video", "note": "man rotates robot"}],
             "confidence": "3", "needs_human_review": False, "review_reason": ""},
            {"episode_id": 2, "start_s": 6.0, "end_s": 8.0, "scenario": "S6",
             "outcome": "pass", "failure_types": ["collision"], "confidence": 2},
        ]}), "mock", {"input_tokens": 0, "output_tokens": 0}


BACKENDS = {"gemini": call_gemini, "claude": call_claude, "mock": call_mock}

# Wait this long before retry 1, 2, 3... Gemini's 503s are fast and RANDOM
# (a text-only "Say OK" failed while a video request right after it worked),
# so many quick retries beat a few long waits. Random jitter is added so
# retries don't line up with everyone else's.
RETRY_WAITS_S = [3, 5, 10, 15, 30, 45, 60, 90]
TRANSIENT = ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded",
             "rate limit", "timed out", "timeout")


def call_with_retries(fn, model, clip, prompt):
    """Retry TEMPORARY errors (server busy, rate limit) with growing waits.

    Permanent errors (bad API key, unknown model) are raised immediately:
    retrying those just wastes time. Growing waits ("backoff") matter because
    if the server is overloaded, hammering it every second makes it worse.
    """
    for attempt in range(len(RETRY_WAITS_S) + 1):
        try:
            return fn(model, clip, prompt)
        except Exception as e:
            msg = str(e)
            if attempt == len(RETRY_WAITS_S) or not any(t.lower() in msg.lower() for t in TRANSIENT):
                raise
            wait = RETRY_WAITS_S[attempt] * random.uniform(0.8, 1.3)
            print(f"   server busy (attempt {attempt + 1}), retrying in {wait:.0f}s")
            time.sleep(wait)
KEYS = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}


# ---------------------------------------------------------------- parsing

def parse_json(raw: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add a sentence; strip that."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def check_response(data: dict, clip_len: float) -> tuple[list[dict], list[str]]:
    episodes, problems = [], []
    for i, ep in enumerate(data.get("episodes", []), 1):
        clean, probs = validate_episode(ep)
        if isinstance(clean["end_s"], float) and clean["end_s"] > clip_len + 1:
            probs.append(f"end_s {clean['end_s']} is past the clip end {clip_len:.1f}")
        problems += [f"episode {i}: {p}" for p in probs]
        clean["validation_problems"] = "; ".join(probs)
        episodes.append(clean)
    return episodes, problems


# ---------------------------------------------------------------- main

def main(backend: str, model: str, only: str | None, force: bool, dry_run: bool,
         tag: str = "") -> None:
    load_env()
    if not dry_run and backend in KEYS and not os.environ.get(KEYS[backend]):
        raise SystemExit(f"No {KEYS[backend]} found. Add a line  {KEYS[backend]}=your-key  "
                         f"to the file .env in this folder, then run again.")

    template = PROMPT_FILE.read_text()
    pv = prompt_version(template)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    if only:
        manifest = manifest[manifest.video_id == only]

    failed = []
    for _, m in manifest.iterrows():
        cp = m.get("content_path") if "content_path" in m else None
        clip = ROOT / cp if isinstance(cp, str) and cp else ROOT / "videos" / "content" / f"{m.video_id}.mp4"
        if not clip.exists():
            print(f"skip  {m.video_id}: no trimmed clip; run segment.py first")
            continue
        offset = float(m.content_start_s) if pd.notna(m.get("content_start_s")) else 0.0
        end = float(m.content_end_s) if pd.notna(m.get("content_end_s")) else float(m.duration_s)
        length = end - offset
        prompt = build_prompt(template, m.video_id, offset, length)

        if dry_run:
            DRY_DIR.mkdir(parents=True, exist_ok=True)
            (DRY_DIR / f"{m.video_id}_prompt.txt").write_text(prompt)
            print(f"dry-run {m.video_id}: prompt -> data/dry_run/{m.video_id}_prompt.txt "
                  f"({len(prompt)} chars), clip {length:.1f}s, "
                  f"{clip.stat().st_size / 1e6:.1f} MB")
            continue

        out = OUT_DIR / f"{m.video_id}__{backend}{'_' + tag if tag else ''}.json"
        if out.exists() and not force:
            print(f"skip  {m.video_id} (already labeled by {backend})")
            continue

        print(f"label {m.video_id} ({m.original_file}) with {backend}:{model} ...")
        t0 = time.time()
        try:
            raw, served_model, usage = call_with_retries(BACKENDS[backend], model, clip, prompt)
        except Exception as e:                        # give up on THIS video, keep going
            print(f"   ✗ gave up: {str(e)[:200]}\n   (no file saved; re-run later to retry it)")
            failed.append(m.video_id)
            continue
        data, problems = None, []
        try:
            data = parse_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            problems = [f"response is not valid JSON: {e}"]
        episodes = []
        if data is not None:
            episodes, problems = check_response(data, length)

        out.write_text(json.dumps({
            "video_id": m.video_id, "backend": backend, "model": model,
            "served_model": served_model, "fps": FPS[backend], "tag": tag,
            # token counts as reported by the API: what you're billed on.
            # Summed over runs, these give the real cost per clip.
            "usage": usage,
            "prompt_version": pv, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "clip_offset_s": offset, "seconds": round(time.time() - t0, 1),
            "summary": (data or {}).get("video_summary", ""),
            "robot_type": (data or {}).get("robot_type", ""),
            "setting": (data or {}).get("setting", ""),
            "episodes": episodes, "problems": problems, "raw_response": raw,
        }, indent=2))

        print(f"   {(data or {}).get('video_summary', '')}")
        for e in episodes:
            flag = "  REVIEW: " + e["review_reason"] if e["needs_human_review"] else ""
            print(f"   ep{e['episode_id']}: {e['start_s']}-{e['end_s']}s  {e['scenario']:<5} "
                  f"{e['outcome']:<8} {','.join(e['failure_types'])}{flag}")
        for p in problems:
            print(f"   ! {p}")
        if not episodes and not problems:
            print("   (no social moments found)")

    if dry_run:
        print("\nOpen the prompt files to see exactly what the model will get. No API calls were made.")
        return

    # Flatten every saved label file into one CSV
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        for e in d["episodes"]:
            row = {"video_id": d["video_id"], "backend": d["backend"], "tag": d.get("tag", ""),
                   "model": d.get("served_model") or d["model"],
                   "prompt_version": d["prompt_version"]}
            row.update({k: e.get(k) for k in EPISODE_FIELDS})
            row["failure_types"] = ",".join(e.get("failure_types") or [])
            row["cf_actions"] = ",".join(e.get("cf_actions") or [])
            row["evidence"] = json.dumps(e.get("evidence") or [])
            row["validation_problems"] = e.get("validation_problems", "")
            rows.append(row)
    pd.DataFrame(rows).to_csv(ALL_CSV, index=False)
    print(f"\nWrote {ALL_CSV} ({len(rows)} episodes)")
    if failed:
        print(f"{len(failed)} video(s) failed this run: {', '.join(failed)}. Run again to retry them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini", choices=BACKENDS)
    ap.add_argument("--model", help="override the default model for the backend")
    ap.add_argument("--only", help="label a single video_id")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fps", type=float, help="frames per second sent to the model (default 2)")
    ap.add_argument("--tag", default="", help="suffix for the output file, to keep experiment runs "
                                               "side by side, e.g. --fps 5 --tag fps5")
    a = ap.parse_args()
    if a.fps:
        FPS[a.backend] = a.fps
    main(a.backend, a.model or DEFAULT_MODELS[a.backend], a.only, a.force, a.dry_run, a.tag)

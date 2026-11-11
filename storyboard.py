"""
Part 4b — Storyboards (what happened, why, what could help)
===========================================================
Turns each labeled clip into a one-page storyboard in three parts:

  A. Chronological storyboard: 6-12 real stills cut from the clip, each with a
     timestamp, a caption of ONLY what is visible, any speech that is heard
     (original + English), and separate "inferred" notes: what we think is
     going on that the footage doesn't show directly, each with why.
  B. What could have happened? Possible reasons for the trouble, each marked
     "seen" (visible in the footage, with a time) or "inferred" (a hypothesis).
  C. What might be missing, or could help prevent this? Design opportunities
     grouped by category.

The VLM DRAFTS the storyboard (frame times, captions, reasons, ideas) from the
clip, the text tracks and its own labels from Part 4. This script then cuts
the stills itself with ffmpeg, so every picture is a real frame from the video,
never a generated image. Like the labels, the draft is a starting point: edit
the JSON and re-run to re-render.

Backends (pick with --backend, same as label.py):
  gemini          sends the video file. Default.
  claude          sends still frames + text tracks.
  mock            no API call: builds a bare storyboard from the labels. For testing.
  claude-session  written by hand in a chat session. Never called; if a file
                  with this backend exists it is used as-is.

Usage:
  python storyboard.py                      # draft missing storyboards, cut stills, render
  python storyboard.py --only 2c78647e87    # one video
  python storyboard.py --render-only        # no API calls; re-cut stills and re-render
  python storyboard.py --fps 1              # send 1 frame/s instead of 2: about half the tokens
  python storyboard.py --force              # draft again with this backend, even if a storyboard exists
                                            # (claude-session files are never overwritten, and still win)

Outputs:
  data/storyboards/<video_id>__<backend>.json   the storyboard (edit freely)
  data/storyboards/frames/<video_id>/*.jpg      the stills
  storyboards.html                              every storyboard on one page; open in your browser
"""

import argparse
import base64
import html
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import label  # reuse its backends, retries, env loading and text tracks

ROOT = Path(__file__).parent
MANIFEST = ROOT / "data" / "manifest.csv"
LABEL_DIR = ROOT / "data" / "labels_vlm"
OUT_DIR = ROOT / "data" / "storyboards"
FRAME_DIR = OUT_DIR / "frames"
PROMPT_FILE = ROOT / "prompts" / "storyboard_prompt.md"
TEMPLATE = ROOT / "storyboard_template.html"
PAGE = ROOT / "storyboards.html"

# Which storyboard to show when a video has several: hand-written first.
PREFER = ["claude-session", "claude", "gemini", "mock"]
CATEGORIES = {
    "communication": ("Clear communication", "comm"),
    "navigation": ("Navigation and space", "nav"),
    "recovery": ("Recovery and help", "rec"),
    "awareness": ("Human awareness", "aware"),
    "environment": ("Environmental design", "env"),
}
MIN_FRAMES, MAX_FRAMES = 6, 12


# ---------------------------------------------------------------- inputs

def label_file(video_id: str) -> Path | None:
    """The labels this storyboard builds on: hand-written first, then newest."""
    files = list(LABEL_DIR.glob(f"{video_id}__*.json"))
    if not files:
        return None
    def rank(f):
        backend = f.stem.split("__", 1)[1]
        return (PREFER.index(backend) if backend in PREFER else len(PREFER), -f.stat().st_mtime)
    return sorted(files, key=rank)[0]


def clip_info(m) -> tuple[Path, float, float, str]:
    """(clip path, offset into the full video, clip length, footage box)."""
    cp = m.get("content_path")
    clip = ROOT / cp if isinstance(cp, str) and cp else ROOT / "videos" / "content" / f"{m.video_id}.mp4"
    offset = float(m.content_start_s) if pd.notna(m.get("content_start_s")) else 0.0
    end = float(m.content_end_s) if pd.notna(m.get("content_end_s")) else float(m.duration_s)
    box = m.get("footage_box") if isinstance(m.get("footage_box"), str) else ""
    return clip, offset, end - offset, box


def labels_as_text(lab: dict) -> str:
    keep = ["episode_id", "start_s", "end_s", "scenario", "scenario_other", "trigger",
            "robot_actions", "robot_speech", "human_actions", "outcome", "failure_types",
            "should_have", "cf_utterance", "evidence"]
    eps = [{k: e.get(k) for k in keep} for e in lab.get("episodes", [])]
    return json.dumps({"summary": lab.get("summary", ""), "setting": lab.get("setting", ""),
                       "episodes": eps}, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- drafting

def mock_storyboard(video_id: str, lab: dict, length: float) -> dict:
    """No model: one frame per labeled evidence moment, reasons from failure types."""
    times = sorted({float(ev["t"]) for e in lab.get("episodes", []) for ev in e.get("evidence") or []
                    if isinstance(ev.get("t"), (int, float)) and 0 <= ev["t"] <= length})
    while len(times) < MIN_FRAMES:                       # pad evenly across the clip
        times = sorted(set(times) | {round(length * (i + 0.5) / MIN_FRAMES, 1) for i in range(MIN_FRAMES)})
    eps = lab.get("episodes", [])
    return {
        "title": f"MOCK storyboard for {video_id}", "summary": lab.get("summary", ""),
        "setting": lab.get("setting", ""), "on_screen_text": "",
        "frames": [{"t_s": t, "caption": "MOCK caption"} for t in times[:MAX_FRAMES]],
        "causes": [{"title": e.get("trigger", "")[:60], "evidence": "inferred", "body": e.get("should_have", "")}
                   for e in eps] or [{"title": "MOCK", "evidence": "inferred", "body": "MOCK"}],
        "opportunities": [{"category": "communication", "title": "MOCK",
                           "items": [e.get("cf_utterance") or "MOCK" for e in eps][:3] or ["MOCK"]}],
    }


def draft(backend: str, model: str, m, lab: dict, template: str) -> tuple[dict | None, str, str, dict, list]:
    clip, offset, length, _ = clip_info(m)
    if backend == "mock":
        return mock_storyboard(m.video_id, lab, length), "", "mock", {}, []
    captions, transcript = label.text_tracks(m.video_id, offset, length)
    prompt = template.format(captions=captions, transcript=transcript, labels=labels_as_text(lab),
                             length=f"{length:.1f}", categories=", ".join(CATEGORIES))
    raw, served, usage = label.call_with_retries(label.BACKENDS[backend], model, clip, prompt)
    try:
        return label.parse_json(raw), raw, served, usage, []
    except (json.JSONDecodeError, ValueError) as e:
        return None, raw, served, usage, [f"response is not valid JSON: {e}"]


def validate(sb: dict, length: float) -> list[str]:
    """Problems a person should look at. The storyboard is still saved and shown."""
    probs = []
    frames = sb.get("frames") or []
    if not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
        probs.append(f"{len(frames)} frames (expected {MIN_FRAMES}-{MAX_FRAMES})")
    last = -1.0
    for i, f in enumerate(frames, 1):
        t = f.get("t_s")
        if not isinstance(t, (int, float)) or not 0 <= t <= length:
            probs.append(f"frame {i}: t_s {t!r} is outside the clip (0-{length:.1f}s)")
        elif t < last:
            probs.append(f"frame {i}: out of order ({t} after {last})")
        else:
            last = t
        if not f.get("caption"):
            probs.append(f"frame {i}: no caption")
        for j, inf in enumerate(f.get("inferred") or [], 1):
            if not isinstance(inf, dict) or not inf.get("claim") or not inf.get("why"):
                probs.append(f"frame {i}: inferred note {j} needs both a claim and a why")
    for i, c in enumerate(sb.get("causes") or [], 1):
        if c.get("evidence") not in ("seen", "inferred"):
            probs.append(f"reason {i}: evidence must be 'seen' or 'inferred', got {c.get('evidence')!r}")
        if c.get("evidence") == "seen" and not c.get("seen_at"):
            probs.append(f"reason {i}: 'seen' needs seen_at (when it is visible)")
    for i, o in enumerate(sb.get("opportunities") or [], 1):
        if o.get("category") not in CATEGORIES:
            probs.append(f"idea group {i}: unknown category {o.get('category')!r}")
    if not sb.get("causes"):
        probs.append("no reasons (part B)")
    if not sb.get("opportunities"):
        probs.append("no design ideas (part C)")
    return probs


# ---------------------------------------------------------------- stills

def parse_box(box: str) -> tuple[int, int, int, int] | None:
    nums = dict(re.findall(r"([xywh])=(\d+)", box or ""))
    return tuple(int(nums[k]) for k in "xywh") if len(nums) == 4 else None


def frame_name(t: float) -> str:
    return f"t{t:07.2f}.jpg"


def cut_stills(video_id: str, clip: Path, sb: dict, box: str) -> list[str]:
    """One JPEG per frame time, cropped to the real footage (or the storyboard's own crop)."""
    out = FRAME_DIR / video_id
    out.mkdir(parents=True, exist_ok=True)
    crop = parse_box(sb.get("crop") or box)
    vf = [f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}"] if crop else []
    vf.append("scale=-2:'min(ih,720)'")
    problems = []
    wanted = {frame_name(f["t_s"]) for f in sb.get("frames") or [] if isinstance(f.get("t_s"), (int, float))}
    for old in out.glob("t*.jpg"):                       # stills for frame times no longer used
        if old.name not in wanted:
            old.unlink()
    for f in sb.get("frames") or []:
        t = f.get("t_s")
        if not isinstance(t, (int, float)):
            continue
        dest = out / frame_name(t)
        if dest.exists() and dest.stat().st_mtime > clip.stat().st_mtime:
            continue                                     # already cut from this version of the clip
        code = subprocess.call(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(clip),
                                "-frames:v", "1", "-vf", ",".join(vf), "-q:v", "4", str(dest)])
        if code != 0 or not dest.exists():
            problems.append(f"could not cut a still at {t:.2f}s")
    return problems


# ---------------------------------------------------------------- page

def clock(t: float) -> str:
    """Seconds as m:ss, keeping one decimal only when there is one (0:07.6)."""
    m, s = divmod(round(t, 1), 60)
    return f"{int(m)}:{s:04.1f}" if s % 1 else f"{int(m)}:{int(s):02d}"


def render_board(video_id: str, sb: dict, lab: dict, m, backend: str) -> str:
    e = html.escape
    frames = []
    for i, f in enumerate(sb.get("frames") or [], 1):
        t = f.get("t_s")
        img = FRAME_DIR / video_id / frame_name(t) if isinstance(t, (int, float)) else None
        src = ("data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()) if img and img.exists() else ""
        when = f.get("t") or (clock(t) if isinstance(t, (int, float)) else "?")
        speech = ""
        if f.get("speech"):
            en = f'<span class="en">{e(f["speech_en"])}</span>' if f.get("speech_en") else ""
            speech = (f'<p class="speech"><span class="lbl">Heard</span>'
                      f'<span class="es">“{e(f["speech"])}”</span>{en}</p>')
        infs = "".join(f'<li><span class="claim">{e(x.get("claim", ""))}</span>'
                       f'<span class="why"><b>Why:</b> {e(x.get("why", ""))}</span></li>'
                       for x in f.get("inferred") or [] if isinstance(x, dict))
        inferred = f'<div class="inferred"><span class="lbl">Inferred</span><ul>{infs}</ul></div>' if infs else ""
        frames.append(f"""
      <li class="frame{' alert' if f.get('alert') else ''}">
        <div class="fhead"><span class="n">{i}</span><span class="tc">{e(when)}</span></div>
        <img src="{src}" alt="Frame at {e(when)}: {e(f.get('caption', ''))}" loading="lazy">
        <p class="cap"><span class="lbl">Seen</span>{e(f.get('caption', ''))}</p>{speech}{inferred}
      </li>""")
    causes = []
    for i, c in enumerate(sb.get("causes") or [], 1):
        seen = c.get("evidence") == "seen"
        tag = f"seen {c.get('seen_at', '')}".strip() if seen else "inferred"
        causes.append(f"""
      <li class="cause">
        <div class="chead"><span class="n">{i}</span><h4>{e(c.get('title', ''))}</h4></div>
        <span class="ev {'seen' if seen else 'inf'}">{e(tag)}</span>
        <p>{e(c.get('body', ''))}</p>
      </li>""")
    opps = []
    for o in sb.get("opportunities") or []:
        name, cls = CATEGORIES.get(o.get("category"), (o.get("category", ""), "comm"))
        items = "".join(f"<li>{e(x)}</li>" for x in o.get("items") or [])
        opps.append(f"""
      <li class="opp {cls}">
        <span class="cat">{e(name)}</span>
        <h4>{e(o.get('title', ''))}</h4>
        <ul>{items}</ul>
      </li>""")
    rows = []
    for ep in lab.get("episodes", []):
        chips = "".join(f'<span class="chip">{e(x)}</span>' for x in ep.get("failure_types") or [])
        cf = f'<p class="cf">“{e(ep["cf_utterance"])}”</p>' if ep.get("cf_utterance") else ""
        sh = f"<p>{e(ep['should_have'])}</p>" if ep.get("should_have") else ""
        rev = f'<p class="vnote">Review: {e(ep["review_reason"])}</p>' if ep.get("needs_human_review") and ep.get("review_reason") else ""
        scen = ep.get("scenario", "") + (f": {ep['scenario_other']}" if ep.get("scenario_other") else "")
        rows.append(f"""
        <li class="ep">
          <div class="vrow"><span class="tc">{e(clock(float(ep.get('start_s') or 0)))}–{e(clock(float(ep.get('end_s') or 0)))}</span>
            <span class="out {e(str(ep.get('outcome', '')))}">{e(str(ep.get('outcome', '')))}</span>{chips}<span class="scen">{e(scen)}</span></div>
          {sh}{cf}{rev}
        </li>""")
    vlm = (f"""
    <aside class="vlm"><p class="vhead">From the labels ({e(lab.get('backend', ''))})</p><ol class="eps">{''.join(rows)}
    </ol></aside>""" if rows else "")
    url = m.get("source_url") if isinstance(m.get("source_url"), str) else ""
    link = f' <a href="{e(url)}" target="_blank" rel="noopener">source</a>' if url else ""
    lang = m.get("language") if isinstance(m.get("language"), str) else ""
    meta = " · ".join(x for x in [video_id, str(m.get("source", "")), clock(float(m.duration_s)), lang] if x)
    probs = sb.get("_problems") or []
    warn = (f'<p class="warn">Check this draft: {e("; ".join(probs))}</p>' if probs else "")
    onscreen = f'<p class="meta"><b>On-screen text</b> {e(sb["on_screen_text"])}</p>' if sb.get("on_screen_text") else ""
    return f"""
  <article class="board" id="v-{video_id}">
    <header class="bhead">
      <p class="eyebrow">{e(meta)}{link} · storyboard by {e(backend)}</p>
      <h2>{e(sb.get('title', video_id))}</h2>
      <p class="summary">{e(sb.get('summary', ''))}</p>
      <p class="meta"><b>Setting</b> {e(sb.get('setting', ''))}</p>{onscreen}{warn}
    </header>
    <section>
      <h3><span class="sl">A</span>Chronological storyboard</h3>
      <p class="sub"><b>Seen</b> describes only what is visible in the still or the stretch of video it covers. <b>Heard</b> is speech from the transcript. <b>Inferred</b> notes are interpretations, each with the reason it was made.</p>
      <div><ol class="frames">{''.join(frames)}
      </ol></div>
    </section>
    <section>
      <h3><span class="sl">B</span>What could have happened?</h3>
      <p class="sub">Possible reasons for the trouble. <span class="ev seen">seen</span> is visible in the footage at that time; <span class="ev inf">inferred</span> is a hypothesis.</p>
      <ol class="causes">{''.join(causes)}
      </ol>
    </section>
    <section>
      <h3><span class="sl">C</span>What might be missing, or could help prevent this?</h3>
      <ul class="opps">{''.join(opps)}
      </ul>{vlm}
    </section>
  </article>"""


def render_page(boards: list[tuple[str, str, str]]) -> None:
    nav = "".join(f'<a href="#v-{vid}"><span class="mono">{vid[:6]}</span>{html.escape(title)}</a>'
                  for vid, title, _ in boards)
    page = TEMPLATE.read_text()
    page = page.replace("{{NAV}}", f'<nav class="picker">{nav}</nav>' if len(boards) > 1 else "")
    page = page.replace("{{BOARDS}}", "".join(b for _, _, b in boards))
    page = page.replace("{{UPDATED}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    PAGE.write_text(page)


def pick_storyboard(video_id: str) -> tuple[Path | None, str]:
    files = {f.stem.split("__", 1)[1]: f for f in OUT_DIR.glob(f"{video_id}__*.json")}
    for b in PREFER + sorted(files):
        if b in files:
            return files[b], b
    return None, ""


# ---------------------------------------------------------------- main

def main(backend: str, model: str, only: str | None, force: bool, render_only: bool) -> None:
    label.load_env()
    if not render_only and backend in label.KEYS and not os.environ.get(label.KEYS[backend]):
        raise SystemExit(f"No {label.KEYS[backend]} found in .env. Use --render-only to just re-render.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template = PROMPT_FILE.read_text()
    pv = label.prompt_version(template)
    manifest = pd.read_csv(MANIFEST)
    if only:
        manifest = manifest[manifest.video_id == only]

    for _, m in manifest.iterrows():
        lf = label_file(m.video_id)
        if lf is None:
            print(f"skip  {m.video_id}: no labels yet; run label.py first")
            continue
        clip, _, length, _ = clip_info(m)
        if not clip.exists():
            print(f"skip  {m.video_id}: no clip; run segment.py first")
            continue
        out = OUT_DIR / f"{m.video_id}__{backend}.json"
        has_one = pick_storyboard(m.video_id)[0] is not None
        if render_only or backend == "claude-session" or (has_one and not force):
            continue                                     # already has a storyboard (any backend)
        print(f"draft {m.video_id} with {backend}:{model} ...")
        t0 = time.time()
        try:
            sb, raw, served, usage, problems = draft(backend, model, m, json.loads(lf.read_text()), template)
        except Exception as ex:                          # give up on THIS video, keep going
            print(f"   ✗ gave up: {str(ex)[:200]} (re-run later to retry)")
            continue
        out.write_text(json.dumps({
            "video_id": m.video_id, "backend": backend, "model": model, "served_model": served,
            "usage": usage, "prompt_version": pv, "labels_from": lf.name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0, 1), "storyboard": sb, "problems": problems,
            "raw_response": raw,
        }, indent=2, ensure_ascii=False))
        print(f"   {(sb or {}).get('title', '(no storyboard)')}")

    # Cut stills and render everything that has a storyboard (not just this run's).
    manifest = pd.read_csv(MANIFEST)
    boards = []
    for _, m in manifest.iterrows():
        sf, used = pick_storyboard(m.video_id)
        lf = label_file(m.video_id)
        if sf is None or lf is None:
            continue
        doc = json.loads(sf.read_text())
        sb = doc.get("storyboard") or {}
        clip, _, length, box = clip_info(m)
        problems = list(doc.get("problems") or []) + validate(sb, length)
        if clip.exists():
            problems += cut_stills(m.video_id, clip, sb, box)
        else:
            problems.append("clip missing, so no stills")
        sb["_problems"] = problems
        for p in problems:
            print(f"   ! {m.video_id}: {p}")
        boards.append((m.video_id, sb.get("title", m.video_id),
                       render_board(m.video_id, sb, json.loads(lf.read_text()), m, used)))
    if not boards:
        print("No storyboards yet.")
        return
    render_page(boards)
    print(f"\nWrote {PAGE.name} ({len(boards)} storyboards, {PAGE.stat().st_size // 1024} KB). Open it in your browser.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini", choices=list(label.BACKENDS) + ["claude-session"])
    ap.add_argument("--model", help="override the default model for the backend")
    ap.add_argument("--only", help="one video_id")
    ap.add_argument("--force", action="store_true", help="redraft (claude-session files are never redrafted)")
    ap.add_argument("--render-only", action="store_true", help="no API calls; re-cut stills and re-render")
    ap.add_argument("--fps", type=float, help="frames per second sent to the model (default 2; 1 halves the cost)")
    a = ap.parse_args()
    if a.fps and a.backend in label.FPS:
        label.FPS[a.backend] = a.fps
    main(a.backend, a.model or label.DEFAULT_MODELS.get(a.backend, ""), a.only, a.force, a.render_only)

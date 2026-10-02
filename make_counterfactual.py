"""
Make a counterfactual video: the original clip, with the robot saying what
it SHOULD have said (cf_utterance), at the moment it should have said it.

Approach "A" from the plan: the footage is untouched; only a synthetic
robot voice and a subtitle are added. So the original and the
counterfactual differ in exactly one thing — whether the robot speaks —
which makes them a clean pair for a user study ("which robot would you
help?"). Physical fixes (stop, back off, yield) can't be shown this way.

The voice is ffmpeg's built-in flite text-to-speech: deliberately robotic.
Every output carries a visible "EDITED" banner; keep these clips internal
(they contain real bystanders) and label them as synthetic anywhere they go.

Usage:
  python make_counterfactual.py --video 2c78647e87                 # first episode with a line to say
  python make_counterfactual.py --video 2c78647e87 --episode 1 --run claude-session
  python make_counterfactual.py --all                               # every eligible episode

Output: videos/counterfactual/<video_id>_ep<N>.mp4  (+ a .json describing the edit)
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "videos" / "counterfactual"
LABELS = ROOT / "data" / "labels_vlm"

# When to speak, relative to the episode (seconds). "on_trigger" waits a
# moment so the trigger is visible first; the "thank" part goes at the end.
TIMING_OFFSET = {"on_trigger": 2.0, "after_waiting": 5.0, "during": 1.0, "after_help": None}
VOICE = "kal16"           # flite voices: kal, kal16, awb, rms, slt
VOICE_GAIN = 2.5          # robot voice louder than the original audio...
DUCK = 0.35               # ...and the original audio turned down while it speaks


def load_run(video_id: str, run: str | None) -> dict:
    files = sorted(LABELS.glob(f"{video_id}__*.json"))
    if run:
        files = [f for f in files if f.stem.split("__", 1)[1] == run]
    runs = [json.loads(f.read_text()) | {"_run": f.stem.split("__", 1)[1]} for f in files]
    runs = [r for r in runs if any(e.get("cf_utterance") for e in r.get("episodes", []))]
    if not runs:
        raise SystemExit(f"No label run for {video_id} has a cf_utterance to voice.")
    return max(runs, key=lambda r: r.get("created", ""))


def plan_lines(ep: dict, clip_len: float) -> list[dict]:
    """Split cf_utterance on ' ... ' and place each part in time."""
    parts = [p.strip() for p in ep["cf_utterance"].split("...") if p.strip()]
    start, end = float(ep["start_s"]), float(ep["end_s"])
    first_off = TIMING_OFFSET.get(ep.get("cf_timing") or "on_trigger")
    t_first = end + 0.5 if first_off is None else start + first_off
    lines = [{"t": round(min(t_first, clip_len - 3), 2), "text": parts[0]}]
    for p in parts[1:]:                     # later parts (usually the thank-you) after the help
        lines.append({"t": round(min(end + 0.5, clip_len - 2), 2), "text": p})
    return lines


def tts(text: str, out: Path) -> float:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"flite=text='{text.replace(chr(39), '')}':voice={VOICE}",
                    "-ar", "48000", "-ac", "2", str(out)], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    return float(dur.strip())


def esc_drawtext(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def wrap(text: str, width: int = 28) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\n".join(lines + [cur])


def render(src: Path, out: Path, lines: list[dict], tmp: Path) -> None:
    inputs, filters, mix = ["-i", str(src)], [], []
    duck_expr = []
    for i, ln in enumerate(lines, 1):
        wav = tmp / f"line{i}.wav"
        ln["dur"] = tts(ln["text"], wav)
        inputs += ["-i", str(wav)]
        delay = int(ln["t"] * 1000)
        filters.append(f"[{i}:a]volume={VOICE_GAIN},adelay={delay}|{delay}[v{i}]")
        mix.append(f"[v{i}]")
        duck_expr.append(f"between(t,{ln['t']},{ln['t'] + ln['dur']})")
    duck = "+".join(duck_expr)
    filters.append(f"[0:a]volume='if({duck},{DUCK},1)':eval=frame[orig]")
    filters.append(f"[orig]{''.join(mix)}amix=inputs={len(lines) + 1}:duration=first:normalize=0[aout]")

    # video: an EDITED banner the whole time + the robot's line as a subtitle while it speaks.
    # The subtitle sits just under the banner (top), because TikTok/Reels captions are burned in near the bottom.
    vf = ["drawbox=x=0:y=0:w=iw:h=34:color=black@0.65:t=fill",
          "drawtext=text='EDITED - robot voice added':x=(w-tw)/2:y=9:"
          "fontsize=min(16\\,w/26):fontcolor=white"]
    for i, ln in enumerate(lines, 1):
        txt = esc_drawtext("ROBOT: " + wrap(ln["text"], 24))
        on = f"between(t,{ln['t']},{ln['t'] + ln['dur'] + 0.6})"
        vf.append(f"drawtext=text='{txt}':x=(w-tw)/2:y=46:fontsize=min(20\\,w/22):fontcolor=white:"
                  f"box=1:boxcolor=0x2a78d6@0.85:boxborderw=12:line_spacing=6:enable='{on}'")
    filters.append(f"[0:v]{','.join(vf)}[vout]")

    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-c:a", "aac", "-b:a", "160k",
                    str(out)], check=True)


def make(video_id: str, episode: int | None, run: str | None) -> list[Path]:
    m = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("video_id").loc[video_id]
    cp = m.get("content_path")
    src = ROOT / (cp if isinstance(cp, str) and cp else m["normalized_path"])
    r = load_run(video_id, run)
    clip_len = float(m.get("content_end_s") or m["duration_s"]) - float(m.get("content_start_s") or 0)
    eps = [e for e in r["episodes"] if e.get("cf_utterance") and (episode is None or e["episode_id"] == episode)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for ep in eps:
        lines = plan_lines(ep, clip_len)
        out = OUT_DIR / f"{video_id}_ep{ep['episode_id']}.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            render(src, out, lines, Path(tmp))
        (out.with_suffix(".json")).write_text(json.dumps({
            "video_id": video_id, "episode_id": ep["episode_id"], "label_run": r["_run"],
            "prompt_version": r.get("prompt_version"), "source_clip": str(src.relative_to(ROOT)),
            "lines": lines, "voice": f"ffmpeg flite {VOICE}",
            "note": "Footage unchanged; synthetic robot voice + subtitle added. Internal use only."},
            indent=2))
        print(f"wrote {out.relative_to(ROOT)}: " + " | ".join(f"{l['t']:.1f}s \"{l['text']}\"" for l in lines))
        made.append(out)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="video_id")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--run", help="label run name, e.g. claude-session or gemini (default: newest)")
    ap.add_argument("--all", action="store_true", help="every video with a cf_utterance")
    a = ap.parse_args()
    if a.all:
        ids = sorted({f.stem.split("__")[0] for f in LABELS.glob("*.json")})
        for vid in ids:
            try:
                make(vid, None, a.run)
            except SystemExit as e:
                print(f"skip {vid}: {e}")
    elif a.video:
        make(a.video, a.episode, a.run)
    else:
        ap.error("give --video <id> or --all")

"""
Part 3b — Transcribe (Whisper)
==============================
Turns the speech in each video into timestamped text:

    {"start_s": 3.2, "end_s": 4.1, "text": "Thank you!", "reliable": true, ...}

This is our EVIDENCE of what was said, including by the robot. Captions
(Part 3a) are the poster's story; this is what the microphone heard.

Whisper's big weakness on social media clips: MUSIC. Given music or noise,
Whisper doesn't say "no speech" — it often invents text ("Thank you for
watching!", lyrics fragments, "♪"). That's dangerous for us: a made-up
"thank you" would turn a robot FAIL into a PASS. So every segment gets a
`reliable` flag, from three checks:
  - no_speech_prob  Whisper's own estimate that there was no speech
  - avg_logprob     how confident it was in the words (closer to 0 = surer)
  - known hallucinations (phrases Whisper famously invents on silence/music)
And each video gets an `audio_type`: speech / little_speech / no_speech.

"no_speech" is NOT proof that nobody spoke. Robots talk quietly; see the
second pass in transcribe(). Treat no_speech as "nothing heard", and have a
person listen whenever a caption claims speech the audio didn't show.

We also use VAD (voice activity detection): a small model that skips
stretches with no human voice BEFORE Whisper sees them, which removes
most hallucinations at the source.

Outputs:
  data/transcripts/<id>.json   segments + video-level summary
  data/transcripts.csv         all videos, one row per segment
  manifest.csv                 gains language, audio_type

Usage (in your Mac Terminal, inside the project folder):
  python3 -m venv .venv && source .venv/bin/activate
  pip install faster-whisper pandas opencv-python
  python transcribe.py                     # 'small' model, good default
  python transcribe.py --model medium      # slower, better on noisy audio
  python transcribe.py --force             # redo all
The first run downloads the model (~500 MB for 'small').
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_DIR = ROOT / "data" / "transcripts"
ALL_CSV = ROOT / "data" / "transcripts.csv"

NO_SPEECH_MAX = 0.6      # above this, Whisper itself thinks there was no speech
LOGPROB_MIN = -1.0       # below this, the words are guesses
NO_SPEECH_LOGPROB = -0.5 # "no speech" only counts if the words are also below this
SPEECH_SHARE_MIN = 0.15  # reliable speech covering <15% of the clip = little_speech
WINDOW_S = 6.0           # pass 2 listens in 6-second windows...
OVERLAP_S = 1.0          # ...overlapping by 1 second
TARGET_RMS = 0.1         # boost quiet stretches toward this loudness (about -20 dBFS)
MAX_GAIN = 15.0          # but never amplify more than 15x (noise gets loud too)

# Phrases Whisper is known to invent on music/silence (lowercase, no punctuation)
HALLUCINATIONS = {
    "thank you for watching", "thanks for watching", "thank you so much for watching",
    "please subscribe", "subscribe to my channel", "like and subscribe",
    "see you in the next video", "subtitles by the amaraorg community",
    "you", "bye", "music", "applause",
}


def normalize(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()


def check(seg) -> tuple[bool, str]:
    """Decide if one Whisper segment is trustworthy; return (reliable, reason)."""
    text = normalize(seg.text)
    if not text or set(seg.text.strip()) <= set("♪♫ .-"):
        return False, "empty_or_music_symbol"
    if text in HALLUCINATIONS:
        return False, "known_hallucination"
    # Whisper's own rule: a segment is "silence" only if no_speech_prob is high
    # AND the words are low-confidence. no_speech_prob alone misfires on
    # fast, slangy speech over street noise: on tt_v004 it was 0.62 for
    # clearly spoken Spanish that Whisper transcribed with high confidence
    # (avg_logprob -0.29). So require both, with a stricter logprob cutoff
    # than Whisper's -1.0 to stay on the cautious side.
    if seg.no_speech_prob > NO_SPEECH_MAX and seg.avg_logprob < NO_SPEECH_LOGPROB:
        return False, "no_speech_prob"
    if seg.avg_logprob < LOGPROB_MIN:
        return False, "low_confidence"
    return True, ""


def boost_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Make quiet speech louder for the second pass (numpy only, no ffmpeg).

    1. Band-pass 250-4000 Hz: keep the frequencies where voices live, drop
       wind rumble and hiss. Done with an FFT: turn the sound into its
       frequencies, zero the ones we don't want, turn it back.
    2. Smart volume knob: measure loudness every 100 ms and raise QUIET
       stretches more than loud ones (up to 15x), so a soft "thank you"
       under street noise comes up to a level Whisper can hear. Gains are
       smoothed so the volume doesn't jump around.
    """
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    spec[(freqs < 250) | (freqs > 4000)] = 0
    x = np.fft.irfft(spec, n=len(audio)).astype(np.float32)

    frame = int(0.1 * sr)
    n = int(np.ceil(len(x) / frame))
    padded = np.pad(x, (0, n * frame - len(x)))
    rms = np.sqrt(np.mean(padded.reshape(n, frame) ** 2, axis=1)) + 1e-6
    gain = np.clip(TARGET_RMS / rms, 1.0, MAX_GAIN)
    gain = np.convolve(gain, np.ones(5) / 5, mode="same")          # smooth over 0.5 s
    per_sample = np.interp(np.arange(len(x)), np.arange(n) * frame + frame / 2, gain)
    return np.clip(x * per_sample, -1.0, 1.0).astype(np.float32)


def seg_to_dict(seg, offset: float, label: str) -> dict:
    reliable, reason = check(seg)
    return {
        "start_s": round(seg.start + offset, 2), "end_s": round(seg.end + offset, 2),
        "text": seg.text.strip(),
        "reliable": reliable, "flag": reason,
        "pass": label,                        # "normal", or "boosted" = found only on pass 2
        "no_speech_prob": round(seg.no_speech_prob, 3),
        "avg_logprob": round(seg.avg_logprob, 3),
        "words": [{"w": w.word.strip(), "s": round(w.start + offset, 2), "e": round(w.end + offset, 2),
                   "p": round(w.probability, 2)} for w in (seg.words or [])],
    }


def pass_normal(model, wav: Path, start: float, end: float):
    """Pass 1: the whole clip, with VAD."""
    segments, info = model.transcribe(
        str(wav),
        vad_filter=True,                      # skip non-voice audio before decoding
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,                 # per-word times, to align with video later
        condition_on_previous_text=False,     # stops one hallucination snowballing into the next
        beam_size=5,
    )
    out = [seg_to_dict(seg, 0.0, "normal") for seg in segments   # decoding happens here
           if seg.start < end and seg.end > start]                # drop outro speech
    return out, info


def pass_boosted(model, wav: Path, start: float, end: float, language: str) -> list[dict]:
    """Pass 2: boosted audio, short windows, no VAD, no silence skipping.

    Three things were hiding the robot's quiet "thank you":
      1. VAD dropped it (not human-sounding enough)      -> VAD off
      2. It was quiet under street noise                  -> boost_audio()
      3. Whisper judges "was there speech?" per 30-SECOND
         window, and skips the whole window if it says no.
         One soft 1-second phrase in 30 s of noise gets
         outvoted by the noise.                            -> 6 s windows, and
                                                              no_speech_threshold=None
    Windows overlap by 1 s so a phrase on a boundary isn't cut in half; each
    segment is kept only by the window whose middle part contains it.
    """
    from faster_whisper.audio import decode_audio
    sr = 16000
    audio = boost_audio(decode_audio(str(wav), sampling_rate=sr), sr)
    out, t = [], start
    while t < end:
        a, b = t, min(t + WINDOW_S, end)
        segments, _ = model.transcribe(
            audio[int(a * sr):int(b * sr)],
            language=language or None,        # short windows detect language badly; reuse pass 1's
            vad_filter=False,
            no_speech_threshold=None,         # never silently skip: show it, flag it, let a human judge
            word_timestamps=True,
            condition_on_previous_text=False,
            beam_size=5,
        )
        lo = a if a == start else a + OVERLAP_S / 2
        hi = b if b >= end else b - OVERLAP_S / 2
        for seg in segments:
            mid = a + (seg.start + seg.end) / 2
            if lo <= mid < hi:
                out.append(seg_to_dict(seg, a, "boosted"))
        if b >= end:
            break
        t = b - OVERLAP_S
    return out


def overlaps(a: dict, b: dict) -> bool:
    return a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]


def transcribe(model, wav: Path, start: float, end: float) -> dict:
    out, info = pass_normal(model, wav, start, end)

    # Robots often speak QUIETLY, in synthetic voices, under street noise, and
    # pass 1 can miss that completely (it did on ig_video_02). So when pass 1
    # heard little or nothing, run pass 2. Without VAD, Whisper also invents
    # more text from noise, so pass-2 results are marked "boosted" and still
    # go through the reliability checks.
    speech_s = sum(s["end_s"] - s["start_s"] for s in out if s["reliable"])
    share = speech_s / max(end - start, 1e-6)
    audio_type = ("no_speech" if speech_s == 0
                  else "little_speech" if share < SPEECH_SHARE_MIN else "speech")

    # Pass 2 results are CANDIDATES for a human to listen to, never evidence.
    # On ig_video_02 pass 2 found the real quiet "thank you" but also invented
    # several more on street noise, often starting exactly at a window edge.
    # So they don't change audio_type, and they're marked reliable=False with
    # flag "listen_check" (or their own failure reason) for the reviewer.
    if share < SPEECH_SHARE_MIN:
        boosted = pass_boosted(model, wav, start, end, info.language)
        for x in boosted:
            if x["reliable"]:
                x["reliable"], x["flag"] = False, "listen_check"
        out += [x for x in boosted if not any(overlaps(x, y) for y in out)]
        out.sort(key=lambda s: s["start_s"])

    return {"language": info.language, "language_prob": round(info.language_probability, 2),
            "audio_type": audio_type, "speech_share": round(share, 2), "segments": out}


def main(model_size: str, force: bool, only: str | None = None) -> None:
    from faster_whisper import WhisperModel   # imported here so --help works without it

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    for col in ("language", "audio_type"):
        if col not in manifest:
            manifest[col] = ""
        manifest[col] = manifest[col].astype(object)

    print(f"loading Whisper '{model_size}' (first time downloads it)...")
    # int8 on CPU: fast on a MacBook and accurate enough for this
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    rows = []
    for i, m in manifest.iterrows():
        out = OUT_DIR / f"{m.video_id}.json"
        if only and m.video_id != only:
            if out.exists():
                result = json.loads(out.read_text())
                rows += [{"video_id": m.video_id, **{k: v for k, v in s.items() if k != "words"}}
                         for s in result["segments"]]
            continue
        if out.exists() and not force:
            print(f"skip  {m.video_id}")
            result = json.loads(out.read_text())
        elif not isinstance(m.audio_path, str) or not m.audio_path:
            print(f"skip  {m.video_id} (no audio track)")
            result = {"language": "", "audio_type": "no_audio", "segments": []}
            out.write_text(json.dumps(result, indent=2))
        else:
            start = m.content_start_s if pd.notna(m.get("content_start_s")) else 0.0
            end = m.content_end_s if pd.notna(m.get("content_end_s")) else m.duration_s
            print(f"whisper {m.video_id} ({m.original_file})")
            result = transcribe(model, ROOT / m.audio_path, float(start), float(end))
            out.write_text(json.dumps(result, indent=2))
            print(f"   language={result['language']}  audio_type={result['audio_type']}"
                  f"  speech_share={result['speech_share']}")
            for s in result["segments"]:
                mark = "  " if s["reliable"] else ("? " if s["flag"] == "listen_check" else "✗ ")
                boost = " (boosted)" if s.get("pass") == "boosted" else ""
                print(f"   {mark}{s['start_s']:6.1f}-{s['end_s']:6.1f}s  {s['text']}{boost}"
                      + (f"   [{s['flag']}]" if s["flag"] else ""))
            if not result["segments"]:
                print("   (no speech detected)")

        manifest.at[i, "language"] = result.get("language", "")
        manifest.at[i, "audio_type"] = result.get("audio_type", "")
        rows += [{"video_id": m.video_id, **{k: v for k, v in s.items() if k != "words"}}
                 for s in result["segments"]]

    manifest.to_csv(MANIFEST, index=False)
    pd.DataFrame(rows).to_csv(ALL_CSV, index=False)
    print(f"\nWrote {ALL_CSV} ({len(rows)} segments; ✗ = unreliable, ? = listen to check)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="(re)transcribe a single video_id")
    a = ap.parse_args()
    main(a.model, a.force or bool(a.only), a.only)

"""
Diagnose Gemini errors by testing ONE thing at a time.

When label.py keeps failing, the question is WHICH part fails:
  1. text only             -> is the API reachable with this key at all?
  2. video, default fps    -> can it take video?
  3. video, custom fps     -> is our fps setting the problem?
  4. video + full prompt   -> is it the size/content of our real request?
Each test runs once (no retries) and reports OK or the exact error.

Usage:  python gemini_check.py [--model gemini-3.8-flash] [--video 9d85a77379]
"""
import argparse
import time
from pathlib import Path

from label import ROOT, build_prompt, PROMPT_FILE, load_env
import os


def run(name, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"  OK    {name}  ({time.time() - t0:.1f}s)  -> {out.strip()[:70]!r}")
    except Exception as e:
        print(f"  FAIL  {name}  ({time.time() - t0:.1f}s)  -> {str(e)[:160]}")


def main(model, video_id):
    load_env()
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    import pandas as pd
    row = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("video_id").loc[video_id]
    cp = row.get("content_path")
    clip = ROOT / cp if isinstance(cp, str) and cp else ROOT / "videos" / "content" / f"{video_id}.mp4"
    blob = types.Blob(data=clip.read_bytes(), mime_type="video/mp4")
    cfg = types.GenerateContentConfig(
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    ask = "Describe this video in one sentence."

    print(f"model={model}  clip={clip.name} ({clip.stat().st_size / 1e6:.1f} MB)")
    run("1 text only", lambda: client.models.generate_content(
        model=model, contents="Say OK.", config=cfg).text)
    run("2 video, default fps", lambda: client.models.generate_content(
        model=model, contents=[types.Part(inline_data=blob), ask], config=cfg).text)
    run("3 video, fps=2", lambda: client.models.generate_content(
        model=model, contents=[types.Part(inline_data=blob,
                                          video_metadata=types.VideoMetadata(fps=2)), ask],
        config=cfg).text)

    import pandas as pd
    m = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("video_id").loc[video_id]
    prompt = build_prompt(PROMPT_FILE.read_text(), video_id, float(m.content_start_s),
                          float(m.content_end_s) - float(m.content_start_s))
    run("4 video + full labeling prompt", lambda: client.models.generate_content(
        model=model, contents=[types.Part(inline_data=blob,
                                          video_metadata=types.VideoMetadata(fps=2)), prompt],
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))).text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--video", default="9d85a77379")
    a = ap.parse_args()
    main(a.model, a.video)

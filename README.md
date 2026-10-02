# HRI video pre-processing pipeline

Turns social media videos of robots into a labeled table: one row per social
moment ("episode"), with the scenario, what the robot did, pass/fail, and what
it should have done instead.

## Run order

| Step | Script | Needs | Output |
|---|---|---|---|
| 0. Fetch | `fetch_videos.py` | yt-dlp | downloads from a sheet/CSV of links, `data/sources.csv` (link + post caption) |
| 1. Ingest | `ingest.py` | ffmpeg | uniform 30 fps copies, audio WAVs, `data/manifest.csv` |
| 2. Segment | `segment.py` | ffmpeg, opencv | camera cuts, outros trimmed, `videos/content/`, contact sheets |
| 3a. Captions | `captions.py` | tesseract, opencv | burned-in text, `data/captions/` |
| 3b. Transcribe | `transcribe.py` | faster-whisper | speech with reliability flags, `data/transcripts/` |
| 4. Label | `label.py` | google-genai or anthropic, API key in `.env` | VLM draft labels, `data/labels_vlm.csv` |
| 5. Compare | `compare.py` (next) | your labels | VLM vs human agreement |

Every script skips videos it has already processed; add `--force` to redo.

**Run everything at once:** `python pipeline.py --sheet "<link>" --batch 10 --all`
runs every step in batches and refreshes the dashboard after each one.

**Dashboard:** `python dashboard.py` writes `dashboard.html` (open it in your browser).
Tabs: Pipeline (videos per stage), Videos (watch each clip with a clickable
timeline of captions, speech and labeled episodes), Dataset (scenarios x
outcomes, failure types, review queue), Problems. Everything stays local.

## Setup (Mac)

```bash
brew install ffmpeg tesseract
python3 -m venv .venv && source .venv/bin/activate
pip install pandas opencv-python faster-whisper google-genai anthropic yt-dlp
cp .env.example .env        # then paste your API key into .env
```

## Files you edit

- `schema.py`: scenarios (S1-S10), outcomes, failure types. The single source of truth.
- `prompts/label_prompt.md`: the instructions the VLM gets. Every label records a
  fingerprint of the prompt version, so edits stay traceable.
- `labels/human_labels.csv`: your hand labels (copy `labels/human_labels_template.csv`).
  Label a clip BEFORE looking at the VLM's output for it, or its answer anchors yours.

## Rules the pipeline follows

- Captions are the poster's story, not evidence of what the robot did.
- Whisper doesn't know who is speaking; the person filming is not the robot.
- "No speech detected" means nothing was heard, not that nothing was said.
  Quiet-audio guesses are marked `listen_check` for a person to verify.
- When sources disagree, the episode is flagged `needs_human_review`.

## Codebook decisions

Rules for ambiguous cases. Each one is also written into `prompts/label_prompt.md`
(so the VLM follows it) and applies to your hand labels. Add new rules here
with a date and the reason; old labels made before a rule may need relabeling.

| # | Date | Rule | Reason |
|---|---|---|---|
| 1 | 2026-09-28 | A stuck/blocked robot that didn't ask for help gets `no_help_request`, even if someone helped without being asked. | The goal is robots that ask for help themselves; unprompted help doesn't excuse not asking. |
| 2 | 2026-10-02 | The counterfactual is the *smallest* behavior change that would make the episode a pass, using only what a wheeled robot can do (speak, stop, move, wait, reroute, call staff). | Keeps fixes realistic for our bell cart and comparable across clips. |
| 3 | 2026-10-02 | Every failure type gets its matching `cf_actions` entry (no_help_request→ask_for_help, no_thanks→thank, no_apology→apologize, no_greeting→greet, failed_to_yield→yield, collision→stop), in the order the robot should do them. | Makes counterfactuals countable and checkable; validation flags mismatches. |
| 4 | 2026-10-02 | `cf_utterance` is one short, polite, specific English line naming what the robot needs; `cf_timing` is for the first action. | Specific requests are actionable for bystanders and usable as robot scripts. |

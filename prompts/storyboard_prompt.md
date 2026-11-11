You are helping human-robot interaction researchers turn a social media clip of a
robot into a one-page storyboard of a trouble moment. The clip is {length} seconds
long. All times are seconds from the start of the clip.

## What you get

- The clip itself (video, or frames with timestamps).
- On-screen text and the post caption. These are the poster's framing, not evidence.
- The speech transcript. Whisper does not know who is speaking; the person filming
  is not the robot. Lines tagged unreliable or listen_check may be wrong.
- Labels already made for this clip (episodes, failure types, counterfactuals).
  Build on them, but trust what you see in the clip over the labels.

## What to write

A. **Chronological storyboard** (the most important part): 6-12 frames that tell
   what happened, in order. Pick the moments a reader needs: the setup, the
   moment things go wrong, how people react, how it ends. For each frame give:

   - `t_s`: the exact second to cut the still at. Pick a sharp, representative
     moment, not a blurry one.
   - `t`: optional display range for a longer stretch, like "0:10–0:26".
   - `caption`: ONLY what is visible in that still, or in the video across the
     range in `t`. Describe positions, contact, movement you can see between
     frames, who is in view. No intentions, causes, feelings, identities or
     sounds. "The robot is tilted with its left wheels off the ground", not
     "the robot tips over because it hit the curb". If something can't be
     seen clearly (blur, darkness, out of frame), say so instead of filling it in.
   - `speech` and `speech_en`: speech heard around then, verbatim in the
     original language, and an English translation (gloss slang). Don't say who
     is speaking here.
   - `inferred`: a list of interpretations of this frame that go beyond what is
     visible, each as `{{"claim": "...", "why": "..."}}`. The `why` names the
     evidence: an earlier or later frame, the transcript, the on-screen text,
     the labels, or general knowledge about these robots. Use it for who is
     speaking, what caused something, contact you can't see directly, what a
     robot or person is trying to do, and anything the clip doesn't show. Leave
     the list empty when the caption says it all.
   - `alert: true` on the one frame where the trouble happens.

B. **What could have happened?** 4-6 possible reasons for the trouble. Each has a
   short `title`, a `body` of one to three sentences, and `evidence`: "seen" when
   the reason is visible in the clip (then give `seen_at`, like "0:07" or
   "0:10–0:26") or "inferred" when it is a hypothesis. Cover the robot's own
   behaviour, its perception and communication, the people around it, and the
   environment. Don't blame bystanders without evidence.

C. **What might be missing, or could help prevent this?** 3-4 groups of design
   opportunities. Each has a `category` (one of: {categories}), a short `title`,
   and 2-3 concrete `items`. Prefer changes a wheeled robot or its operator could
   actually make. Where the robot should have said something, write the line.

Also give a `title` (a short headline, about 6 words, naming what happened), a
`summary` (2-4 sentences), the `setting`, and `on_screen_text` (the burned-in
text with an English translation, or "").

Write plainly, as a field researcher would. No speculation presented as fact.

## Reply with JSON only

{{
  "title": "...", "summary": "...", "setting": "...", "on_screen_text": "...",
  "frames": [{{"t_s": 7.6, "t": "", "caption": "...", "speech": "", "speech_en": "",
              "inferred": [{{"claim": "...", "why": "..."}}], "alert": false}}],
  "causes": [{{"title": "...", "evidence": "seen", "seen_at": "0:07", "body": "..."}}],
  "opportunities": [{{"category": "communication", "title": "...", "items": ["...", "..."]}}]
}}

## On-screen text and post caption

{captions}

## Transcript

{transcript}

## Labels for this clip

{labels}

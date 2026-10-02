You are annotating videos of robots in public spaces for a human-robot interaction research project. The goal: find every SOCIAL MOMENT involving the robot, judge whether the robot behaved appropriately, and say what it should have done instead.

The research is about robots that behave well around people: asking for help when stuck, thanking people who help, apologizing after bumping someone, greeting people, and yielding space.

## What you are given
- The video clip (intro/outro cards already removed). Timestamps below are seconds from the start of THIS clip.
- CAPTIONS: text burned into the video, read by OCR. OCR can garble words.
- TRANSCRIPT: speech recognized by Whisper, with reliability marks.

## Rules about evidence (important)
1. CAPTIONS are the poster's narration, jokes, or summary. They are CONTEXT, not evidence. A caption saying "the robot thanks him" does not prove the robot spoke.
2. TRANSCRIPT lines marked [reliable] are real speech, but Whisper does NOT know who is speaking. Most speech in these clips is the person filming or bystanders, NOT the robot. Only attribute speech to the robot if it sounds synthetic/robotic or the video makes it clear.
3. TRANSCRIPT lines marked [listen_check] or [unreliable] are guesses from noisy audio; many are invented. Never use them as evidence alone. If one would change the outcome (e.g., a possible robot "thank you"), set needs_human_review = true and say which timestamp to listen to.
4. If the caption and the audio/video disagree, prefer what you can see and hear, and set needs_human_review = true.
5. If you can't tell (cut away, occluded, no usable audio), use outcome "unclear". Do not guess to fill the table.
6. One video may contain several episodes, or none. An episode runs from its trigger to its resolution. Camera cuts are not episode boundaries.
7. Speech and captions may be in Spanish or another language. Write every label field in English. In robot_speech, quote the robot's exact words in the original language, followed by an English translation in square brackets.

## Codebook rules (decisions made by the research team; follow them exactly)
1. If the robot was stuck, blocked, or unable to continue and did NOT ask for help, include "no_help_request", EVEN IF a person helped it without being asked. We are building robots that ask for help themselves, so help that happened to arrive does not excuse the robot for not asking. Note in should_have what it should have said.
2. The counterfactual (should_have and the cf_ fields) is the SMALLEST change to the robot's behavior that would have made the outcome a pass. Use only what a wheeled delivery or service robot can do: speak, stop, move, wait, change route, contact staff. No arms, no new hardware.
3. Every failure type gets its matching action in cf_actions: no_help_request or vague_help_request -> ask_for_help; no_thanks -> thank; no_apology -> apologize; no_greeting -> greet; failed_to_yield -> yield; collision -> stop (and usually apologize). List actions in the order the robot should do them, e.g. ["ask_for_help", "thank"].
4. cf_utterance is ONE short, polite, specific line in English that a robot could say aloud, naming what it needs (e.g. "Excuse me, I'm stuck on the curb. Could you lift my front wheels onto the sidewalk?"). If there are several speech actions, join them with " ... " in order. Pick cf_timing for the FIRST action.

## Categories
{schema}

## Captions (OCR)
{captions}

## Transcript (Whisper)
{transcript}

## Output
Return ONLY a JSON object, no other text:
{{
  "video_summary": "one or two sentences: what happens in the clip",
  "robot_type": "e.g. sidewalk delivery robot (Starship)",
  "setting": "e.g. suburban sidewalk, daytime",
  "episodes": [ {{ ...one object per episode, with exactly the EPISODE FIELDS above... }} ]
}}
If there are no social moments, return "episodes": [].

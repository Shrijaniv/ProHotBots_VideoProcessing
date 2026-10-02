"""
The label schema: what one labeled EPISODE looks like.

This file is the single source of truth for label fields. label.py (the VLM),
your hand labels (labels/human_labels_template.csv), and compare.py (Part 5)
all use these exact names and allowed values. If you change a category here,
change it once, and everything else follows.

An EPISODE = one social moment: from the trigger (robot gets stuck, a person
approaches, contact happens) to the resolution (helped, person leaves, or the
clip ends unresolved). One video can hold several episodes.
"""

# Scenarios from the project doc (Part 1 scenario catalog). Add new ones here.
SCENARIOS = {
    "S1": "Closed door: door in path, robot cannot open it",
    "S2": "Elevator: robot needs a floor, cannot press buttons",
    "S3": "Blocked hallway: person or group standing in the robot's path",
    "S4": "Stuck: wedged, tipped, or disoriented; no progress (incl. after a turn)",
    "S5": "Unclimbable step: curb or step with no ramp",
    "S6": "Contact: robot bumps into or drives into a person",
    "S7": "Approach: a person walks up to or engages the robot",
    "S8": "Passing: robot and person pass in a narrow space",
    "S9": "Ignored: robot asks for help and nobody responds",
    "S10": "Busy people: the only nearby people are occupied (phone, conversation)",
    "other": "A social moment not covered above (describe it in scenario_other)",
}

OUTCOMES = {
    "pass": "robot behaved appropriately for the moment",
    "fail": "robot missed the appropriate social behavior",
    "partial": "right idea, poor execution (e.g., thanked but inaudibly, asked but vaguely)",
    "unclear": "can't tell from this clip (cut away, occluded, audio missing)",
}

# Failure types: pick all that apply for fail/partial; empty list for pass.
FAILURE_TYPES = {
    "no_help_request": "was stuck/blocked and did not ask for help, INCLUDING when "
                       "someone helped before it asked (codebook rule 1)",
    "vague_help_request": "asked, but not specifically enough to act on",
    "no_thanks": "was helped but did not thank",
    "no_apology": "made contact / inconvenienced someone but did not apologize",
    "no_greeting": "ignored a person who engaged it",
    "collision": "made physical contact with a person",
    "failed_to_yield": "did not give way / kept personal distance",
    "resisted_human_guidance": "fought a person physically redirecting it",
    "too_quiet": "spoke, but too quietly for the setting",
    "inappropriate_speech": "said something wrong, rude, or badly timed",
    "other": "anything else (explain in should_have)",
}

# ---- Structured counterfactual: what the robot SHOULD have done (level 1)
# Free text ("should_have") is hard to count; these fields make it countable:
# "in 40 stuck episodes the fix was ask_for_help, addressed to the nearest
# person, on_trigger". Codebook rules 2-4 in the prompt explain how to fill them.
CF_ACTIONS = {
    "ask_for_help": "ask a person for a specific kind of help",
    "thank": "thank the person who helped",
    "apologize": "apologize to the person affected",
    "greet": "greet / acknowledge a person engaging with it",
    "announce": "say what it is doing or about to do (\"passing on your left\")",
    "stop": "stop moving",
    "back_off": "move away / create space",
    "yield": "wait and let people pass first",
    "reroute": "take a different path",
    "escalate_to_staff": "notify an operator / staff when nobody helps",
    "other": "anything else (describe in should_have)",
}
CF_ADDRESSEES = {
    "nearest_person": "the closest available person",
    "helper": "the person who helped",
    "affected_person": "the person bumped, blocked, or inconvenienced",
    "approaching_person": "the person engaging with the robot",
    "group": "a group of people",
    "staff": "an operator or staff member",
    "none": "no one (e.g. a purely physical action like stop)",
}
CF_TIMINGS = {
    "on_trigger": "immediately, within ~2 s of the trigger",
    "after_waiting": "after briefly trying / waiting first (e.g. stuck for a few seconds)",
    "after_help": "once the help is complete",
    "during": "while the interaction is happening",
}
# Which counterfactual action each failure type implies (used for consistency checks)
FAILURE_TO_ACTION = {
    "no_help_request": "ask_for_help", "vague_help_request": "ask_for_help",
    "no_thanks": "thank", "no_apology": "apologize", "no_greeting": "greet",
    "failed_to_yield": "yield", "collision": "stop",
}

# Every episode has exactly these fields.
EPISODE_FIELDS = {
    "episode_id": "int, 1..N in time order",
    "start_s": "float, seconds from the start of the clip",
    "end_s": "float",
    "scenario": "one of the SCENARIOS keys",
    "scenario_other": "short text if scenario == 'other', else ''",
    "trigger": "what started the moment",
    "robot_actions": "what the robot physically did",
    "robot_speech": "exact words the ROBOT said, in quotes, in the original language with an English translation in [brackets]; 'none' if it said nothing; 'inaudible' if it seems to speak but can't be made out",
    "human_actions": "what the people did",
    "outcome": "one of OUTCOMES",
    "failure_types": "list of FAILURE_TYPES keys; [] if outcome is pass",
    "should_have": "what the robot should have done, one sentence; '' if pass",
    "cf_actions": "list of CF_ACTIONS keys in the order the robot should do them; [] if pass",
    "cf_utterance": "the exact words the robot should say, in English; '' if no speech is needed",
    "cf_addressee": "one of CF_ADDRESSEES (who it should speak to); 'none' if no speech; '' if pass",
    "cf_timing": "one of CF_TIMINGS (when it should act); '' if pass",
    "evidence": "list of {t: seconds, source: video|audio|caption, note: text}",
    "confidence": "1 (guess) | 2 (probable) | 3 (clearly visible/audible)",
    "needs_human_review": "true/false",
    "review_reason": "why a human should look, else ''",
}

EVIDENCE_SOURCES = {"video", "audio", "caption"}


def validate_episode(ep: dict) -> tuple[dict, list[str]]:
    """Check one episode against the schema. Returns (cleaned episode, problems).

    Models sometimes return near-misses ("Pass", "s5", confidence "3").
    We fix what's unambiguous and report everything else, so a bad label is
    never silently accepted.
    """
    problems = []
    out = {}
    missing = [f for f in EPISODE_FIELDS if f not in ep]
    if missing:
        problems.append(f"missing fields: {missing}")
    for f in EPISODE_FIELDS:
        out[f] = ep.get(f)

    sc = str(out["scenario"] or "").strip()
    sc = sc.upper() if sc.lower() != "other" else "other"
    if sc not in SCENARIOS:
        problems.append(f"unknown scenario {out['scenario']!r}")
    out["scenario"] = sc

    oc = str(out["outcome"] or "").strip().lower()
    if oc not in OUTCOMES:
        problems.append(f"unknown outcome {out['outcome']!r}")
    out["outcome"] = oc

    ft = out["failure_types"] or []
    if isinstance(ft, str):
        ft = [x.strip() for x in ft.replace(";", ",").split(",") if x.strip()]
    ft = [x.strip().lower() for x in ft]
    bad = [x for x in ft if x not in FAILURE_TYPES]
    if bad:
        problems.append(f"unknown failure_types {bad}")
    if oc == "pass" and ft:
        problems.append("outcome is pass but failure_types is not empty")
    if oc in ("fail", "partial") and not ft:
        problems.append(f"outcome is {oc} but failure_types is empty")
    out["failure_types"] = ft

    for f in ("start_s", "end_s"):
        try:
            out[f] = round(float(out[f]), 2)
        except (TypeError, ValueError):
            problems.append(f"{f} is not a number")
    if isinstance(out["start_s"], float) and isinstance(out["end_s"], float) \
            and out["end_s"] < out["start_s"]:
        problems.append("end_s before start_s")

    try:
        out["confidence"] = int(out["confidence"])
        if out["confidence"] not in (1, 2, 3):
            raise ValueError
    except (TypeError, ValueError):
        problems.append(f"confidence must be 1, 2 or 3, got {out['confidence']!r}")

    nhr = out["needs_human_review"]
    if isinstance(nhr, str):
        nhr = nhr.strip().lower() in ("true", "yes", "1")
    out["needs_human_review"] = bool(nhr)

    ev = out["evidence"] or []
    if not isinstance(ev, list):
        problems.append("evidence is not a list"); ev = []
    for e in ev:
        if not isinstance(e, dict) or e.get("source") not in EVIDENCE_SOURCES:
            problems.append(f"bad evidence item {e!r}")
    out["evidence"] = ev

    # ---- structured counterfactual
    acts = out["cf_actions"] or []
    if isinstance(acts, str):
        acts = [x.strip() for x in acts.replace(";", ",").split(",") if x.strip()]
    acts = [x.strip().lower() for x in acts]
    bad = [a for a in acts if a not in CF_ACTIONS]
    if bad:
        problems.append(f"unknown cf_actions {bad}")
    out["cf_actions"] = acts
    for f, allowed in (("cf_addressee", CF_ADDRESSEES), ("cf_timing", CF_TIMINGS)):
        v = str(out[f] or "").strip().lower()
        if v and v not in allowed:
            problems.append(f"unknown {f} {out[f]!r}")
        out[f] = v
    if oc == "pass" and (acts or out["cf_timing"]):
        problems.append("outcome is pass but a counterfactual is filled in")
    if oc in ("fail", "partial"):
        if not acts:
            problems.append(f"outcome is {oc} but cf_actions is empty")
        if not out["cf_timing"]:
            problems.append(f"outcome is {oc} but cf_timing is empty")
        # each failure type should be fixed by its matching action
        for f_type in ft:
            need = FAILURE_TO_ACTION.get(f_type)
            if need and need not in acts:
                problems.append(f"failure {f_type} but cf_actions lacks {need}")
        speaks = any(a in acts for a in ("ask_for_help", "thank", "apologize", "greet", "announce"))
        if speaks and not str(out["cf_utterance"] or "").strip():
            problems.append("cf_actions include speech but cf_utterance is empty")

    for f in ("scenario_other", "trigger", "robot_actions", "robot_speech",
              "human_actions", "should_have", "review_reason", "cf_utterance"):
        out[f] = "" if out[f] is None else str(out[f])
    return out, problems


def schema_as_text() -> str:
    """The schema written out for the prompt, so the model sees the exact rules."""
    lines = ["SCENARIOS:"] + [f"  {k}: {v}" for k, v in SCENARIOS.items()]
    lines += ["", "OUTCOMES:"] + [f"  {k}: {v}" for k, v in OUTCOMES.items()]
    lines += ["", "FAILURE_TYPES:"] + [f"  {k}: {v}" for k, v in FAILURE_TYPES.items()]
    lines += ["", "CF_ACTIONS:"] + [f"  {k}: {v}" for k, v in CF_ACTIONS.items()]
    lines += ["", "CF_ADDRESSEES:"] + [f"  {k}: {v}" for k, v in CF_ADDRESSEES.items()]
    lines += ["", "CF_TIMINGS:"] + [f"  {k}: {v}" for k, v in CF_TIMINGS.items()]
    lines += ["", "EPISODE FIELDS:"] + [f"  {k}: {v}" for k, v in EPISODE_FIELDS.items()]
    return "\n".join(lines)

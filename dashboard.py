"""
Dashboard — see the dataset being built
=======================================
Reads everything the pipeline has written in data/ and generates ONE file,
dashboard.html, in the project folder. Open it in your browser.

  python dashboard.py            # regenerate (pipeline.py does this for you)
  open dashboard.html            # Mac: opens it in your default browser

Everything stays on your computer: the page is a plain file that plays the
videos straight from videos/ next to it. Nothing is uploaded anywhere, which
matters while the IRB/data-handling question is open (clips show bystanders).

Tabs:
  Pipeline  how many videos have reached each stage
  Videos    every video and its progress; click one to watch it with a
            timeline of captions, speech and labeled episodes (click any row
            to jump the video to that moment)
  Dataset   scenarios x outcomes, most common failure types, review queue
  Problems  failed downloads and label-validation problems
"""

import csv
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "dashboard.html"

STAGES = [
    ("listed", "In link list"),
    ("downloaded", "Downloaded"),
    ("ingested", "Ingested"),
    ("segmented", "Segmented"),
    ("captions", "Captions (OCR)"),
    ("transcribed", "Transcribed"),
    ("labeled", "VLM-labeled"),
    ("human", "Hand-labeled"),
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def counterfactuals(vid: str) -> list[dict]:
    """Edited clips from make_counterfactual.py: original footage + robot voice."""
    out = []
    for j in sorted((ROOT / "videos" / "counterfactual").glob(f"{vid}_ep*.json")):
        mp4 = j.with_suffix(".mp4")
        meta = read_json(j) or {}
        if mp4.exists():
            out.append({"episode": meta.get("episode_id"), "video": str(mp4.relative_to(ROOT)),
                        "lines": meta.get("lines", []), "run": meta.get("label_run", "")})
    return out


def collect() -> dict:
    manifest = read_csv(DATA / "manifest.csv")
    sources = read_csv(DATA / "sources.csv")
    links = read_csv(DATA / "links.csv")
    failures = read_csv(DATA / "download_failures.csv")
    shots = read_csv(DATA / "shots.csv")
    human = read_csv(ROOT / "labels" / "human_labels.csv")

    src_by_file = {r["file"]: r for _, r in sources.iterrows()} if not sources.empty else {}
    fetched_urls = set(sources["url"]) if not sources.empty else set()

    # --- VLM labels: newest file per (video, backend+tag)
    labels: dict[str, list] = {}
    for f in sorted((DATA / "labels_vlm").glob("*.json")):
        d = read_json(f)
        if not d:
            continue
        labels.setdefault(d["video_id"], []).append({
            "run": f.stem.split("__", 1)[1] if "__" in f.stem else d.get("backend", ""),
            "backend": d.get("backend", ""), "model": d.get("served_model") or d.get("model", ""),
            "prompt_version": d.get("prompt_version", ""), "created": d.get("created", ""),
            "fps": d.get("fps", ""), "summary": d.get("summary", ""),
            "robot_type": d.get("robot_type", ""), "setting": d.get("setting", ""),
            "episodes": d.get("episodes", []), "problems": d.get("problems", []),
        })
    for v in labels.values():
        v.sort(key=lambda r: r["created"], reverse=True)

    human_ids = set(human["video_id"]) if not human.empty and "video_id" in human else set()

    videos = []
    for _, m in manifest.iterrows():
        vid = m["video_id"]
        src = src_by_file.get(m.get("original_file", ""), {})
        offset = num(m.get("content_start_s"), 0.0) or 0.0
        cap = read_json(DATA / "captions" / f"{vid}.json")
        tr = read_json(DATA / "transcripts" / f"{vid}.json")
        content_path = m.get("content_path") or ""
        if not content_path:
            p = ROOT / "videos" / "content" / f"{vid}.mp4"
            content_path = str(p.relative_to(ROOT)) if p.exists() else m.get("normalized_path", "")
        # the player shows the content clip, so shift text timestamps into clip time
        uses_trim = "videos/content/" in content_path
        shift = offset if uses_trim else 0.0
        vlabels = labels.get(vid, [])
        latest = vlabels[0] if vlabels else None
        episodes = latest["episodes"] if latest else []
        vshots = shots[shots["video_id"] == vid] if not shots.empty else pd.DataFrame()

        videos.append({
            "id": vid,
            "file": m.get("original_file", ""),
            "sheet_id": src.get("sheet_id", ""),
            "platform": m.get("source", ""),
            "url": m.get("source_url", "") or src.get("url", ""),
            "post_caption": m.get("post_caption", "") or src.get("caption", ""),
            "duration": num(m.get("duration_s"), 0.0),
            "content_start": num(m.get("content_start_s")),
            "content_end": num(m.get("content_end_s")),
            "language": m.get("language", "") or (tr or {}).get("language", ""),
            "audio_type": m.get("audio_type", "") or (tr or {}).get("audio_type", ""),
            "video": content_path,
            "contact": f"data/contact/{vid}.jpg" if (DATA / "contact" / f"{vid}.jpg").exists() else "",
            "notes": m.get("notes", ""),
            "stages": {
                "listed": True,
                "downloaded": True,
                "ingested": True,
                "segmented": bool(m.get("n_shots")),
                "captions": cap is not None,
                "transcribed": tr is not None,
                "labeled": bool(vlabels),
                "human": vid in human_ids,
            },
            "shots": [{"start": num(r["start_s"]) - shift, "end": num(r["end_s"]) - shift,
                       "kind": r["kind"]} for _, r in vshots.iterrows()],
            "captions": [{"start": c["start_s"] - shift, "end": c["end_s"] - shift,
                          "text": c["text"], "conf": c.get("conf")} for c in (cap or [])],
            "speech": [{"start": s["start_s"] - shift, "end": s["end_s"] - shift, "text": s["text"],
                        "reliable": s["reliable"], "flag": s.get("flag", ""), "pass": s.get("pass", "normal")}
                       for s in (tr or {}).get("segments", [])],
            "runs": vlabels,
            "episodes": episodes,
            "counterfactuals": counterfactuals(vid),
            "needs_review": any(e.get("needs_human_review") for e in episodes),
        })

    # --- link list: all links ever seen (links.csv), else sources + failures
    all_links = {}
    for df in (links, sources, failures):
        if not df.empty and "url" in df:
            for _, r in df.iterrows():
                all_links.setdefault(r["url"], r.get("sheet_id", ""))
    downloaded_urls = fetched_urls
    hand_added = sum(1 for v in videos if not v["url"])

    open_failures = []
    if not failures.empty:
        latest_fail = failures.drop_duplicates("url", keep="last")
        for _, r in latest_fail.iterrows():
            if r["url"] not in downloaded_urls:
                open_failures.append({"sheet_id": r.get("sheet_id", ""), "url": r["url"],
                                      "error": r.get("error", ""), "at": r.get("at", "")})

    counts = {k: sum(1 for v in videos if v["stages"][k]) for k, _ in STAGES}
    counts["listed"] = len(all_links) + hand_added
    counts["downloaded"] = len(downloaded_urls) + hand_added

    problems = []
    for v in videos:
        for run in v["runs"]:
            for p in run["problems"]:
                problems.append({"video": v["id"], "file": v["file"], "run": run["run"], "problem": p})

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stages": [{"key": k, "label": lbl, "count": counts[k]} for k, lbl in STAGES],
        "videos": videos,
        "failures": open_failures,
        "problems": problems,
        "pending_links": [{"sheet_id": s, "url": u} for u, s in all_links.items()
                          if u not in downloaded_urls and u not in {f["url"] for f in open_failures}],
    }


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Video Dataset</title>
<style>
:root{
  --bg:#f6f6f4;--surface:#fcfcfb;--line:#e4e3de;--text:#0b0b0b;--text-2:#52514e;--muted:#8a8984;
  --accent:#2a78d6;--accent-soft:#e8f1fc;
  --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--neutral:#a3a29c;
  --lane-cap:#2a78d6;--lane-speech:#1baf7a;--lane-ep:#4a3aa7;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#121211;--surface:#1a1a19;--line:#33332f;--text:#ffffff;--text-2:#c3c2b7;--muted:#8d8c85;
  --accent:#3987e5;--accent-soft:#1d2a3b;--neutral:#6b6a64;
  --lane-cap:#3987e5;--lane-speech:#199e70;--lane-ep:#9085e9;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--line:#33332f;--text:#ffffff;--text-2:#c3c2b7;--muted:#8d8c85;
  --accent:#3987e5;--accent-soft:#1d2a3b;--neutral:#6b6a64;--lane-cap:#3987e5;--lane-speech:#199e70;--lane-ep:#9085e9;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;padding:18px 24px 0}
h1{font-size:20px;margin:0;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--text-2);font-size:13px}
.spacer{flex:1}
label.auto{color:var(--text-2);font-size:13px;display:flex;gap:6px;align-items:center}
nav{display:flex;gap:4px;padding:12px 24px 0;border-bottom:1px solid var(--line)}
.cfb{border:1px solid var(--border,#ccc);background:none;border-radius:999px;padding:3px 10px;font:inherit;font-size:12px;color:var(--text-2);cursor:pointer}
.cfb.on{background:#2a78d6;border-color:#2a78d6;color:#fff}
nav button{background:none;border:0;border-bottom:2px solid transparent;padding:8px 12px;font:inherit;color:var(--text-2);cursor:pointer}
nav button.on{color:var(--text);border-bottom-color:var(--accent);font-weight:600}
main{padding:20px 24px 40px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px}
.panel h2{font-size:14px;margin:0 0 4px;font-weight:650}
.panel .lead{color:var(--text-2);font-size:13px;margin:0 0 14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .n{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .l{color:var(--text-2);font-size:12px}
.bars{display:grid;grid-template-columns:150px 1fr 70px;gap:8px 12px;align-items:center}
.bars .lbl{color:var(--text-2);font-size:13px;text-align:right}
.bars .track{height:18px;position:relative}
.bars .bar{height:100%;background:var(--accent);border-radius:0 4px 4px 0;min-width:2px}
.bars .val{font-variant-numeric:tabular-nums;font-size:13px}
.bars .row:hover .bar{filter:brightness(1.1)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--text-2);font-weight:600;font-size:12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.filters input,.filters select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--text)}
.filters input{flex:1;min-width:180px}
.layout{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:16px;align-items:start}
@media (max-width:900px){.layout{grid-template-columns:1fr}.bars{grid-template-columns:110px 1fr 50px}}
.list{max-height:calc(100vh - 230px);overflow:auto;padding:0}
.item{padding:10px 14px;border-bottom:1px solid var(--line);cursor:pointer}
.item:hover{background:var(--accent-soft)}
.item.sel{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
.item .t{display:flex;gap:8px;align-items:baseline}
.item .t b{font-weight:600}
.item .meta{color:var(--muted);font-size:12px}
.dots{display:flex;gap:3px;margin-top:6px}
.dot{width:14px;height:6px;border-radius:2px;background:var(--line)}
.dot.on{background:var(--accent)}
.chip{display:inline-flex;align-items:center;gap:4px;font-size:12px;padding:1px 7px;border-radius:999px;border:1px solid var(--line);color:var(--text-2);white-space:nowrap}
.chip i{width:8px;height:8px;border-radius:50%;display:inline-block}
.o-pass i{background:var(--good)} .o-partial i{background:var(--warning)} .o-fail i{background:var(--critical)} .o-unclear i{background:var(--neutral)}
.review{color:var(--critical);font-size:12px;font-weight:600}
.detail video{width:100%;max-height:60vh;background:#000;border-radius:8px}
.cols{display:grid;grid-template-columns:minmax(220px,340px) 1fr;gap:16px}
@media (max-width:1200px){.cols{grid-template-columns:1fr}}
.kv{display:grid;grid-template-columns:110px 1fr;gap:4px 10px;font-size:13px}
.kv div:nth-child(odd){color:var(--text-2)}
.timeline{position:relative;margin:8px 0 4px}
.lane{display:grid;grid-template-columns:70px 1fr;align-items:center;gap:8px;margin:4px 0}
.lane span{color:var(--text-2);font-size:12px;text-align:right}
.track2{position:relative;height:16px;background:var(--bg);border-radius:4px}
.seg{position:absolute;top:2px;bottom:2px;border-radius:3px;cursor:pointer;min-width:3px}
.seg.cap{background:var(--lane-cap)} .seg.sp{background:var(--lane-speech)} .seg.sp.bad{background:none;border:1px dashed var(--lane-speech)}
.seg.ep{background:var(--lane-ep)} .seg.shot{background:var(--neutral);opacity:.5} .seg.shot.static{opacity:.25}
.playhead{position:absolute;top:0;bottom:0;width:2px;background:var(--text);pointer-events:none}
.axis{display:grid;grid-template-columns:70px 1fr;gap:8px;color:var(--muted);font-size:11px}
.axis div:last-child{display:flex;justify-content:space-between}
.rows{max-height:300px;overflow:auto}
.rows table td:first-child{white-space:nowrap;color:var(--accent);cursor:pointer;font-variant-numeric:tabular-nums}
.ep{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
.ep h3{margin:0 0 6px;font-size:13px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.ep .kv{grid-template-columns:120px 1fr}
.muted{color:var(--muted)}
a{color:var(--accent)}
.tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);font-size:12px;padding:6px 8px;border-radius:6px;max-width:360px;z-index:10;display:none}
.heat td.num{min-width:54px}
.heat td.c0{color:var(--muted)}
.empty{color:var(--muted);padding:20px 0}
.tabhide{display:none}
</style></head>
<body>
<header>
  <h1>Robot Video Dataset</h1>
  <span class="sub">Generated __GENERATED__ · re-run <code>python dashboard.py</code> (or pipeline.py) to update</span>
  <span class="spacer"></span>
  <label class="auto"><input type="checkbox" id="auto"> Auto-reload every 30 s</label>
</header>
<nav id="tabs">
  <button data-t="pipeline" class="on">Pipeline</button>
  <button data-t="videos">Videos</button>
  <button data-t="dataset">Dataset</button>
  <button data-t="problems">Problems</button>
</nav>
<main>
  <section id="pipeline"></section>
  <section id="videos" class="tabhide"></section>
  <section id="dataset" class="tabhide"></section>
  <section id="problems" class="tabhide"></section>
</main>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = (s, el=document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = t => { t = Math.max(0, t||0); const m = Math.floor(t/60), s = (t%60).toFixed(1).padStart(4,'0'); return `${m}:${s}`; };
const STAGE_KEYS = D.stages.map(s => s.key);
const tip = $('#tip');
function showTip(e, text){ tip.textContent = text; tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px'; }
function hideTip(){ tip.style.display='none'; }

// ---------------- tabs + auto reload (remembered per browser)
function setTab(t){
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.t===t));
  document.querySelectorAll('main > section').forEach(s => s.classList.toggle('tabhide', s.id!==t));
  try{ localStorage.setItem('rvd-tab', t); }catch(e){}
}
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => setTab(b.dataset.t));
let timer=null;
$('#auto').onchange = e => { try{ localStorage.setItem('rvd-auto', e.target.checked?'1':''); }catch(_){} armAuto(); };
function armAuto(){ clearInterval(timer); if ($('#auto').checked) timer = setInterval(() => location.reload(), 30000); }

// ---------------- helpers
function outcomeChip(o){ return o ? `<span class="chip o-${esc(o)}"><i></i>${esc(o)}</span>` : ''; }
function barRows(rows, total){
  const max = Math.max(1, ...rows.map(r => r.count));
  return `<div class="bars">` + rows.map(r => `
    <div class="lbl">${esc(r.label)}</div>
    <div class="track row" data-tip="${esc(r.label)}: ${r.count}${total?` of ${total}`:''}"><div class="bar" style="width:${(r.count/max*100).toFixed(1)}%"></div></div>
    <div class="val">${r.count}${total?` <span class="muted">(${Math.round(r.count/Math.max(total,1)*100)}%)</span>`:''}</div>`).join('') + `</div>`;
}
function wireTips(root){
  root.querySelectorAll('[data-tip]').forEach(el => { el.onmousemove = e => showTip(e, el.dataset.tip); el.onmouseleave = hideTip; });
}

// ---------------- Pipeline tab
function renderPipeline(){
  const V = D.videos, eps = V.flatMap(v => v.episodes);
  const total = D.stages[0].count;
  const review = V.filter(v => v.needs_review).length;
  $('#pipeline').innerHTML = `
    <div class="tiles">
      <div class="tile"><div class="n">${V.length}</div><div class="l">videos in the pipeline</div></div>
      <div class="tile"><div class="n">${(V.reduce((a,v)=>a+(v.duration||0),0)/60).toFixed(1)}</div><div class="l">minutes of video</div></div>
      <div class="tile"><div class="n">${eps.length}</div><div class="l">labeled episodes</div></div>
      <div class="tile"><div class="n">${review}</div><div class="l">videos flagged for human review</div></div>
      <div class="tile"><div class="n">${D.failures.length}</div><div class="l">failed downloads (unresolved)</div></div>
    </div>
    <div class="panel"><h2>Videos at each stage</h2>
      <p class="lead">Each bar counts videos that have finished that stage. Gaps show where work is waiting.</p>
      ${barRows(D.stages, total)}
    </div>
    <div class="panel"><h2>Next steps</h2>${nextSteps()}</div>`;
  wireTips($('#pipeline'));
}
function nextSteps(){
  const c = Object.fromEntries(D.stages.map(s => [s.key, s.count])), out = [];
  const add = (n, what, cmd) => n > 0 && out.push(`<tr><td class="num">${n}</td><td>${what}</td><td><code>${cmd}</code></td></tr>`);
  add(D.pending_links.length, 'links not downloaded yet', 'python fetch_videos.py --sheet "…" --user-agent ""');
  add(c.ingested - c.segmented, 'need segmenting', 'python segment.py');
  add(c.ingested - c.captions, 'need captions (OCR)', 'python captions.py');
  add(c.ingested - c.transcribed, 'need transcribing', 'python transcribe.py');
  add(c.ingested - c.labeled, 'need VLM labels', 'python label.py');
  add(c.ingested - c.human, 'not hand-labeled yet', 'labels/human_labels.csv');
  return out.length ? `<table><tr><th class="num">Videos</th><th>Waiting for</th><th>Command</th></tr>${out.join('')}</table>`
                    : '<p class="empty">Everything is up to date.</p>';
}

// ---------------- Videos tab
let sel = null;
function renderVideos(){
  $('#videos').innerHTML = `
    <div class="layout">
      <div class="panel" style="padding:12px">
        <div class="filters">
          <input id="q" placeholder="Search file, ID, caption, label…">
          <select id="fStage"><option value="">Any stage</option>${D.stages.slice(2).map(s=>`<option value="${s.key}">Missing: ${esc(s.label)}</option>`).join('')}</select>
          <select id="fOut"><option value="">Any outcome</option><option>pass</option><option>partial</option><option>fail</option><option>unclear</option><option value="review">Needs review</option></select>
        </div>
        <div class="list" id="list"></div>
      </div>
      <div class="panel detail" id="detail"><p class="empty">Pick a video on the left.</p></div>
    </div>`;
  ['q','fStage','fOut'].forEach(id => $('#'+id).oninput = drawList);
  drawList();
  const first = D.videos.find(v => v.id === sel) || D.videos[0];
  if (first) select(first.id);
}
function matches(v){
  const q = $('#q').value.toLowerCase().trim(), st = $('#fStage').value, o = $('#fOut').value;
  if (st && v.stages[st]) return false;
  if (o === 'review' && !v.needs_review) return false;
  if (o && o !== 'review' && !v.episodes.some(e => e.outcome === o)) return false;
  if (!q) return true;
  const hay = [v.id, v.file, v.sheet_id, v.post_caption, ...v.captions.map(c=>c.text), ...v.episodes.flatMap(e=>[e.scenario,e.trigger,e.should_have,(e.failure_types||[]).join(' ')])].join(' ').toLowerCase();
  return hay.includes(q);
}
function drawList(){
  const items = D.videos.filter(matches);
  $('#list').innerHTML = items.length ? items.map(v => `
    <div class="item ${v.id===sel?'sel':''}" data-id="${v.id}">
      <div class="t"><b>${esc(v.sheet_id || v.file)}</b><span class="meta">${esc(v.platform)} · ${fmt(v.duration)}${v.language?` · ${esc(v.language)}`:''}</span></div>
      <div class="meta">${esc(v.file)}</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px">${v.episodes.map(e => outcomeChip(e.outcome)).join('')}${v.needs_review?'<span class="review">⚑ review</span>':''}</div>
      <div class="dots">${STAGE_KEYS.map(k => `<span class="dot ${v.stages[k]?'on':''}" data-tip="${esc(D.stages.find(s=>s.key===k).label)}: ${v.stages[k]?'done':'not yet'}"></span>`).join('')}</div>
    </div>`).join('') : '<p class="empty" style="padding:14px">No videos match.</p>';
  $('#list').querySelectorAll('.item').forEach(el => el.onclick = () => select(el.dataset.id));
  wireTips($('#list'));
}
function select(id){
  sel = id; const v = D.videos.find(x => x.id === id); if (!v) return;
  document.querySelectorAll('.item').forEach(el => el.classList.toggle('sel', el.dataset.id === id));
  const run = v.runs[0];
  const len = Math.max(0.1, ...[v.content_end && v.content_start!=null ? v.content_end - (v.video.includes('videos/content/')?v.content_start:0) : 0, v.duration||0,
                              ...v.speech.map(s=>s.end), ...v.captions.map(c=>c.end), ...v.episodes.map(e=>e.end_s||0)].filter(x=>x>0));
  const segs = (arr, cls, label, startK='start', endK='end') => arr.map((s,i) => {
    const a = Math.max(0, s[startK]), b = Math.min(len, Math.max(s[endK], a+0.1));
    return `<div class="seg ${cls(s)}" style="left:${a/len*100}%;width:${(b-a)/len*100}%" data-t="${a}" data-tip="${esc(label(s))}"></div>`; }).join('');
  $('#detail').innerHTML = `
    <div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px">
      <h2 style="margin:0;font-size:16px">${esc(v.sheet_id || v.file)}</h2>
      <span class="muted">${esc(v.id)}</span>
      ${v.url ? `<a href="${esc(v.url)}" target="_blank" rel="noopener">Open original ↗</a>` : ''}
    </div>
    <div class="cols">
      <div>
        ${v.counterfactuals.length ? `<div class="cfswitch" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px">
          <button class="cfb on" data-src="${esc(v.video)}">Original</button>
          ${v.counterfactuals.map(c=>`<button class="cfb" data-src="${esc(c.video)}" data-lines="${esc(c.lines.map(l=>fmt(l.t)+' “'+l.text+'”').join('  ·  '))}">Counterfactual · ep ${esc(c.episode)}</button>`).join('')}
        </div>` : ''}
        <video id="player" src="${esc(v.video)}" controls preload="metadata"></video>
        <p id="cfnote" class="muted" style="margin:4px 0 0;display:none"></p>
        <div class="kv" style="margin-top:10px">
          <div>File</div><div>${esc(v.file)}</div>
          <div>Language</div><div>${esc(v.language||'—')}</div>
          <div>Audio</div><div>${esc(v.audio_type||'—')}</div>
          <div>Post caption</div><div>${esc(v.post_caption||'—')}</div>
          ${run?`<div>Labeled by</div><div>${esc(run.model)} · prompt ${esc(run.prompt_version)} · ${esc(run.created.slice(0,16).replace('T',' '))}</div>`:''}
          ${v.notes?`<div>Notes</div><div>${esc(v.notes)}</div>`:''}
        </div>
        ${v.contact?`<details style="margin-top:10px"><summary class="muted">Contact sheet (1 frame per second)</summary><img src="${esc(v.contact)}" style="width:100%;margin-top:6px;border-radius:6px" loading="lazy"></details>`:''}
      </div>
      <div>
        <h2 style="font-size:14px;margin:0">Timeline</h2>
        <p class="lead" style="margin:2px 0 6px">Hover a block for its text; click to jump the video there. Dashed speech = flagged unreliable.</p>
        <div class="timeline" id="tl">
          <div class="lane"><span>Episodes</span><div class="track2">${segs(v.episodes, e=>'ep', e=>`Episode ${e.episode_id}: ${e.scenario} · ${e.outcome} — ${e.trigger}`, 'start_s','end_s')}</div></div>
          <div class="lane"><span>Captions</span><div class="track2">${segs(v.captions, ()=>'cap', c=>c.text)}</div></div>
          <div class="lane"><span>Speech</span><div class="track2">${segs(v.speech, s=>'sp'+(s.reliable?'':' bad'), s=>`${s.text}${s.reliable?'':` [${s.flag}]`}`)}</div></div>
          <div class="lane"><span>Shots</span><div class="track2">${segs(v.shots, s=>'shot '+(s.kind==='static'?'static':''), s=>`${s.kind} shot ${fmt(s.start)}–${fmt(s.end)}`)}</div></div>
          <div class="axis"><div></div><div><span>0:00</span><span>${fmt(len/2)}</span><span>${fmt(len)}</span></div></div>
        </div>
        <h2 style="font-size:14px;margin:14px 0 4px">Labeled episodes ${run?`<span class="muted" style="font-weight:400">(${run.episodes.length})</span>`:''}</h2>
        ${run ? (run.summary?`<p class="lead">${esc(run.summary)}</p>`:'') + (run.episodes.length ? run.episodes.map(epCard).join('') : '<p class="empty">No social moments found.</p>')
              : '<p class="empty">Not labeled yet. Run <code>python label.py</code>.</p>'}
        ${v.runs.length>1?`<details><summary class="muted">${v.runs.length-1} earlier/other label run(s)</summary>${v.runs.slice(1).map(r=>`<p class="muted">${esc(r.run)} · ${esc(r.model)} · prompt ${esc(r.prompt_version)}: ${r.episodes.map(e=>`${e.scenario} ${e.outcome}`).join(', ')||'no episodes'}</p>`).join('')}</details>`:''}
        <h2 style="font-size:14px;margin:14px 0 4px">Text tracks</h2>
        <div class="rows"><table>
          <tr><th>Time</th><th>Source</th><th>Text</th></tr>
          ${[...v.captions.map(c=>({t:c.start,src:'caption',text:c.text,ok:true})), ...v.speech.map(s=>({t:s.start,src:'speech'+(s.pass==='boosted'?' (boosted)':''),text:s.text,ok:s.reliable,flag:s.flag}))]
            .sort((a,b)=>a.t-b.t).map(r=>`<tr><td data-t="${Math.max(0,r.t)}">${fmt(r.t)}</td><td class="muted">${esc(r.src)}</td><td>${esc(r.text)}${r.ok?'':` <span class="muted">[${esc(r.flag)}]</span>`}</td></tr>`).join('') || '<tr><td colspan="3" class="muted">None yet.</td></tr>'}
        </table></div>
      </div>
    </div>`;
  const player = $('#player');
  const jump = t => { player.currentTime = +t; player.play().catch(()=>{}); };
  $('#detail').querySelectorAll('[data-t]').forEach(el => el.onclick = () => jump(el.dataset.t));
  const tl = $('#tl');
  player.ontimeupdate = () => {
    tl.querySelectorAll('.playhead').forEach(p => p.remove());
    tl.querySelectorAll('.track2').forEach(tr => { const p = document.createElement('div'); p.className='playhead'; p.style.left=(player.currentTime/len*100)+'%'; tr.appendChild(p); });
  };
  $('#detail').querySelectorAll('.cfb').forEach(b => b.onclick = () => {
    const t = player.currentTime;
    $('#detail').querySelectorAll('.cfb').forEach(x => x.classList.toggle('on', x===b));
    player.src = b.dataset.src;
    player.addEventListener('loadedmetadata', () => { player.currentTime = t; }, {once:true});
    const note = $('#cfnote');
    note.style.display = b.dataset.lines ? '' : 'none';
    note.textContent = b.dataset.lines ? 'Edited clip (synthetic robot voice, internal only). Robot says: ' + b.dataset.lines : '';
  });
  wireTips($('#detail'));
}
function epCard(e){
  return `<div class="ep">
    <h3><span data-t="${e.start_s||0}" style="color:var(--accent);cursor:pointer">${fmt(e.start_s)}–${fmt(e.end_s)}</span>
      <span>${esc(e.scenario)}${e.scenario_other?`: ${esc(e.scenario_other)}`:''}</span>${outcomeChip(e.outcome)}
      ${(e.failure_types||[]).map(f=>`<span class="chip">${esc(f)}</span>`).join('')}
      <span class="muted">confidence ${esc(e.confidence)}</span>
      ${e.needs_human_review?`<span class="review">⚑ review: ${esc(e.review_reason)}</span>`:''}</h3>
    <div class="kv">
      <div>Trigger</div><div>${esc(e.trigger)}</div>
      <div>Robot did</div><div>${esc(e.robot_actions)}</div>
      <div>Robot said</div><div>${esc(e.robot_speech)}</div>
      <div>People did</div><div>${esc(e.human_actions)}</div>
      ${e.should_have?`<div>Should have</div><div>${esc(e.should_have)}</div>`:''}
      ${(e.cf_actions||[]).length?`<div>Fix (actions)</div><div>${e.cf_actions.map(a=>`<span class="chip">${esc(a)}</span>`).join(' → ')} <span class="muted">${esc(e.cf_timing||'')}${e.cf_addressee&&e.cf_addressee!=='none'?` · to ${esc(e.cf_addressee)}`:''}</span></div>`:''}
      ${e.cf_utterance?`<div>Should say</div><div>“${esc(e.cf_utterance)}”</div>`:''}
      ${e.validation_problems?`<div>Validation</div><div class="review">${esc(e.validation_problems)}</div>`:''}
    </div></div>`;
}

// ---------------- Dataset tab
function renderDataset(){
  const eps = D.videos.flatMap(v => v.episodes.map(e => ({...e, video:v})));
  const outcomes = ['pass','partial','fail','unclear'];
  const scen = [...new Set(eps.map(e => e.scenario))].sort((a,b) => (a==='other')-(b==='other') || a.localeCompare(b, undefined, {numeric:true}));
  const cnt = (s,o) => eps.filter(e => e.scenario===s && e.outcome===o).length;
  const ft = {}; eps.forEach(e => (e.failure_types||[]).forEach(f => ft[f]=(ft[f]||0)+1));
  const ftRows = Object.entries(ft).sort((a,b)=>b[1]-a[1]).map(([label,count])=>({label,count}));
  const review = eps.filter(e => e.needs_human_review);
  const cf = {}; eps.forEach(e => (e.cf_actions||[]).forEach(a => cf[a]=(cf[a]||0)+1));
  const cfRows = Object.entries(cf).sort((a,b)=>b[1]-a[1]).map(([label,count])=>({label,count}));
  // stuck/blocked scenarios (S1-S5); how many were labeled "didn't ask for help"
  const stuck = eps.filter(e => ['S1','S2','S3','S4','S5'].includes(e.scenario));
  const noAsk = stuck.filter(e => (e.failure_types||[]).includes('no_help_request'));
  $('#dataset').innerHTML = eps.length ? `
    <div class="tiles">
      <div class="tile"><div class="n">${eps.length}</div><div class="l">episodes from ${D.videos.filter(v=>v.episodes.length).length} videos</div></div>
      ${outcomes.map(o=>`<div class="tile"><div class="n">${eps.filter(e=>e.outcome===o).length}</div><div class="l">${outcomeChip(o)}</div></div>`).join('')}
      <div class="tile"><div class="n">${noAsk.length} / ${stuck.length}</div><div class="l">stuck/blocked episodes labeled <b>no_help_request</b> (labels made before codebook rule 1 may undercount)</div></div>
    </div>
    <div class="panel"><h2>Scenarios × outcomes</h2><p class="lead">Episode counts from the latest VLM labels (drafts until reviewed).</p>
      <table class="heat"><tr><th>Scenario</th>${outcomes.map(o=>`<th class="num">${outcomeChip(o)}</th>`).join('')}<th class="num">Total</th></tr>
      ${scen.map(s=>`<tr><td>${esc(s)}</td>${outcomes.map(o=>{const n=cnt(s,o);return `<td class="num c${n?1:0}">${n||'·'}</td>`}).join('')}<td class="num"><b>${eps.filter(e=>e.scenario===s).length}</b></td></tr>`).join('')}
      </table></div>
    <div class="panel"><h2>Most common failure types</h2><p class="lead">Across all failed or partial episodes.</p>
      ${ftRows.length ? barRows(ftRows) : '<p class="empty">No failures labeled yet.</p>'}</div>
    <div class="panel"><h2>Counterfactual fixes</h2><p class="lead">What the robot should have done instead, counted across failed or partial episodes.</p>
      ${cfRows.length ? barRows(cfRows) : '<p class="empty">No structured counterfactuals yet (labels from before 2026-10-02 only have free text).</p>'}</div>
    <div class="panel"><h2>Review queue</h2><p class="lead">Episodes the model flagged for a person to check.</p>
      ${review.length ? `<table><tr><th>Video</th><th>Time</th><th>Scenario</th><th>Why</th></tr>${review.map(e=>`<tr><td><a href="#" data-open="${e.video.id}">${esc(e.video.sheet_id||e.video.file)}</a></td><td>${fmt(e.start_s)}</td><td>${esc(e.scenario)} ${outcomeChip(e.outcome)}</td><td>${esc(e.review_reason)}</td></tr>`).join('')}</table>` : '<p class="empty">Nothing flagged.</p>'}
    </div>` : '<div class="panel"><p class="empty">No labeled episodes yet. Run <code>python label.py</code>.</p></div>';
  $('#dataset').querySelectorAll('[data-open]').forEach(a => a.onclick = ev => { ev.preventDefault(); sel = a.dataset.open; setTab('videos'); renderVideos(); });
  wireTips($('#dataset'));
}

// ---------------- Problems tab
function renderProblems(){
  $('#problems').innerHTML = `
    <div class="panel"><h2>Downloads that failed</h2><p class="lead">Only links that never succeeded. They're retried automatically on the next fetch run.</p>
      ${D.failures.length ? `<table><tr><th>ID</th><th>Link</th><th>Last error</th><th>When</th></tr>${D.failures.map(f=>`<tr><td>${esc(f.sheet_id)}</td><td><a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.url.split('/').pop())}</a></td><td>${esc(f.error.slice(0,180))}</td><td class="muted">${esc(f.at.slice(0,16).replace('T',' '))}</td></tr>`).join('')}</table>` : '<p class="empty">None.</p>'}</div>
    <div class="panel"><h2>Label validation problems</h2><p class="lead">Fields the model got wrong (unknown category, missing field, pass with failures…). These labels need a closer look.</p>
      ${D.problems.length ? `<table><tr><th>Video</th><th>Run</th><th>Problem</th></tr>${D.problems.map(p=>`<tr><td>${esc(p.file)}</td><td class="muted">${esc(p.run)}</td><td>${esc(p.problem)}</td></tr>`).join('')}</table>` : '<p class="empty">None.</p>'}</div>
    <div class="panel"><h2>Links not downloaded yet</h2>
      ${D.pending_links.length ? `<p class="lead">${D.pending_links.length} link(s) waiting.</p>` : '<p class="empty">None waiting.</p>'}</div>`;
}

renderPipeline(); renderVideos(); renderDataset(); renderProblems();
try{ setTab(localStorage.getItem('rvd-tab') || 'pipeline'); $('#auto').checked = !!localStorage.getItem('rvd-auto'); }catch(e){}
armAuto();
</script></body></html>
"""


def main():
    data = collect()
    blob = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", blob).replace("__GENERATED__", html.escape(data["generated"]))
    OUT.write_text(page, encoding="utf-8")
    c = {s["key"]: s["count"] for s in data["stages"]}
    print(f"Wrote {OUT.name}: {len(data['videos'])} videos, "
          f"{sum(len(v['episodes']) for v in data['videos'])} episodes, "
          f"{c['labeled']} labeled, {len(data['failures'])} open download failures")


if __name__ == "__main__":
    main()

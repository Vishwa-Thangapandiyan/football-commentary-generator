"""
Live demo: pick a verified test-split event, generate commentary from the model.

Tiny local web UI, Python standard library only - no Flask, no Streamlit, no CDN,
no internet. Loads the trained checkpoint once at startup and serves one page.

    python demo_live.py
    -> open http://127.0.0.1:8000

Does NOT touch the model or the training pipeline. Inference only, using the same
constrained decoding as generate.py (CLAUDE.md 4.14).
"""

import html
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch

from generate import build_banned_mask, load_model, restore_names
from train import CommentaryDataset, collate

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "full_best.pt"
DATA = ROOT / "data" / "processed.json"
PORT = 8000
N_EVENTS = 40          # dropdown size; kept small so the list stays readable

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading {CKPT.name} on {DEVICE} ...")
MODEL, VOCAB, CK = load_model(str(CKPT), DEVICE)
EXAMPLES = json.loads(DATA.read_text(encoding="utf-8"))["test"]
print(f"loaded. {len(EXAMPLES):,} official test examples available.")


def pick_events():
    """A varied, verified set: grounded players + a real event label.
    Index 84 (the Herrera goal used in the rendered demo) is pinned first."""
    chosen, seen = [], set()
    if EXAMPLES[84]["label"] == "soccer-ball":
        chosen.append(84)
        seen.add(84)
    wanted = ["soccer-ball", "y-card", "substitution", "corner", "injury",
              "penalty", "r-card", "soccer-ball-own", "penalty-missed", "yr-card"]
    # round-robin over event types so the dropdown is not all corners
    per_type = {w: [] for w in wanted}
    for i, e in enumerate(EXAMPLES):
        if i in seen or not e["grounding"]:
            continue
        if e["label"] in per_type and len(per_type[e["label"]]) < 8:
            per_type[e["label"]].append(i)
    for rank in range(8):
        for w in wanted:
            if rank < len(per_type[w]) and len(chosen) < N_EVENTS:
                chosen.append(per_type[w][rank])
    return chosen


EVENT_IDS = pick_events()
print(f"dropdown: {len(EVENT_IDS)} verified events")


@torch.no_grad()
def generate_one(idx):
    ex = EXAMPLES[idx]
    ds = CommentaryDataset([ex], VOCAB)
    src, _ = collate([ds[0]])
    src = src.to(DEVICE)
    banned = build_banned_mask([ex], VOCAB, DEVICE)
    out = MODEL.greedy_decode(src, max_len=CK["max_tgt"], banned=banned)
    tagged = VOCAB.decode(out[0].tolist())
    return {
        "label": ex["label"],
        "gameTime": ex["gameTime"],
        "game": ex["game"].replace("\\", " / "),
        "input": ex["input"],
        "grounding": {k: v["name"] for k, v in ex["grounding"].items()},
        "generated_tagged": tagged,
        "generated": restore_names(tagged, ex["grounding"]),
        "reference": restore_names(ex["target"], ex["grounding"]),
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Football Commentary Generator</title><style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;padding:24px;
background:#12151a;color:#e8eaed;line-height:1.5}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#9aa0a6;font-size:13px;margin-bottom:20px}
select,button{font-size:15px;padding:9px 12px;border-radius:6px;border:1px solid #3c4043}
select{background:#1e2127;color:#e8eaed;min-width:520px}
button{background:#1a73e8;color:#fff;border:none;cursor:pointer;margin-left:8px}
button:disabled{background:#5f6368;cursor:default}
.card{background:#1e2127;border:1px solid #3c4043;border-radius:8px;
padding:14px 16px;margin-top:16px}
.k{color:#9aa0a6;font-size:12px;text-transform:uppercase;letter-spacing:.5px;
margin-bottom:4px}
.mono{font-family:Consolas,monospace;font-size:13px;color:#8ab4f8;
word-break:break-word}
.gen{font-size:17px;color:#81c995}
.ref{color:#bdc1c6}
.note{margin-top:22px;padding:12px 14px;background:#2a2118;border-left:3px solid
#f9ab00;border-radius:4px;font-size:13px;color:#fdd663}
</style></head><body><div class="wrap">
<h1>Football Commentary Generator</h1>
<div class="sub">Custom multi-head-attention Transformer &middot; greedy decoding
&middot; official SoccerNet test split (never seen in training)</div>
<select id="ev">__OPTIONS__</select><button id="go" onclick="run()">Generate</button>
<div id="out"></div>
<div class="note"><b>Scope:</b> the event type and player identities come from the
SoccerNet annotation. The Transformer generates the commentary text. Player names
are never in the model's vocabulary &mdash; it emits
<span class="mono">&lt;PLAYER_1&gt;</span> slots that are filled in afterwards, so it
cannot invent a player.</div>
</div><script>
function esc(s){var d=document.createElement('div');d.innerText=s;return d.innerHTML}
function run(){
 var b=document.getElementById('go');b.disabled=true;b.innerText='Generating...';
 var i=document.getElementById('ev').value;
 fetch('/api/generate?idx='+i).then(r=>r.json()).then(d=>{
  document.getElementById('out').innerHTML=
   '<div class="card"><div class="k">Match</div>'+esc(d.game)+
   '<br><span class="mono">'+esc(d.label)+' &middot; '+esc(d.gameTime)+'</span></div>'+
   '<div class="card"><div class="k">Structured input to the encoder</div>'+
   '<div class="mono">'+esc(d.input)+'</div></div>'+
   '<div class="card"><div class="k">Grounded players</div><div class="mono">'+
   esc(JSON.stringify(d.grounding))+'</div></div>'+
   '<div class="card"><div class="k">Generated (raw model output)</div>'+
   '<div class="mono">'+esc(d.generated_tagged)+'</div></div>'+
   '<div class="card"><div class="k">Generated commentary</div>'+
   '<div class="gen">'+esc(d.generated)+'</div></div>'+
   '<div class="card"><div class="k">Human reference</div>'+
   '<div class="ref">'+esc(d.reference)+'</div></div>';
  b.disabled=false;b.innerText='Generate';
 }).catch(e=>{document.getElementById('out').innerHTML=
   '<div class="card">Error: '+esc(String(e))+'</div>';
  b.disabled=false;b.innerText='Generate';});
}
</script></body></html>"""


def build_options():
    opts = []
    for i in EVENT_IDS:
        e = EXAMPLES[i]
        who = ", ".join(v["name"] or "?" for v in e["grounding"].values())
        teams = e["game"].split("\\")[-1]
        label = f"{e['label']}  |  {e['gameTime']}  |  {who[:44]}  |  {teams[:40]}"
        opts.append(f'<option value="{i}">{html.escape(label)}</option>')
    return "".join(opts)


OPTIONS_HTML = build_options()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, PAGE.replace("__OPTIONS__", OPTIONS_HTML),
                       "text/html; charset=utf-8")
        elif u.path == "/api/generate":
            try:
                idx = int(parse_qs(u.query).get("idx", ["84"])[0])
                if not 0 <= idx < len(EXAMPLES):
                    raise ValueError("index out of range")
                payload = json.dumps(generate_one(idx), ensure_ascii=False)
                self._send(200, payload, "application/json; charset=utf-8")
            except Exception as exc:
                self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                           "application/json; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass          # keep the console clean during the demo


if __name__ == "__main__":
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Demo ready at {url}   (Ctrl+C to stop)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")

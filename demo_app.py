"""
Video-first demo: the match plays, and as the playhead reaches each action-spotting
detection, the Transformer generates that commentary LIVE and it appears in the feed.

    python demo_app.py
    -> http://127.0.0.1:8000

What is precomputed vs live:
  * PRECOMPUTED - the Spivak action-spotting pass over the half (it is a batch model
    over 224-frame chunks, not a streaming one). Cached in
    outputs/spivak_detections_half1.json.
  * LIVE        - every commentary line. The Transformer runs greedy decoding at the
    moment the video reaches the event. The page shows the server-side generation
    time in milliseconds so you can see it is computed, not replayed.

Nothing here trains or modifies the Transformer, its checkpoint or its vocabulary.

Requires outputs/web/half1.mp4 (see --remux, needs ffmpeg once).
"""

import json
import os
import random
import re
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch

from generate import build_banned_mask, load_model, restore_names
from sampling_experiment import sample_decode      # isolated, reusable sampler
from spot_to_commentary import build_input
from train import CommentaryDataset, collate

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "outputs" / "web" / "half1.mp4"
MKV = (ROOT / "data" / "caption-2024" / "england_epl" / "2014-2015" /
       "2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal" / "1_224p.mkv")
DETECTIONS = ROOT / "outputs" / "spivak_detections_half1.json"
CKPT = ROOT / "checkpoints" / "full_best.pt"
PORT = 8000
HOME, AWAY = "Manchester United", "Arsenal"
SHOW = {"soccer-ball", "corner"}     # classes that read cleanly without a player

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ensure_video():
    if VIDEO.exists():
        return
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    print("remuxing mkv -> mp4 (stream copy) ...")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(MKV),
                    "-c", "copy", "-movflags", "+faststart", str(VIDEO)], check=True)
    print(f"   {VIDEO} ({VIDEO.stat().st_size/1e6:.0f} MB)")


ensure_video()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading Transformer on {DEVICE} ...")
MODEL, VOCAB, CK = load_model(str(CKPT), DEVICE)

EVENTS = [e for e in json.loads(DETECTIONS.read_text(encoding="utf-8"))
          if e["label"] in SHOW]
EVENTS.sort(key=lambda e: e["t"])
print(f"loaded {len(EVENTS)} spivak detections ({sorted(SHOW)})")


@torch.no_grad()
def generate_for(idx):
    ev = EVENTS[idx]
    src_text = build_input(ev, HOME, AWAY)
    ex = {"input": src_text, "target": "", "grounding": {}}
    ds = CommentaryDataset([ex], VOCAB)
    src, _ = collate([ds[0]])
    banned = build_banned_mask([ex], VOCAB, DEVICE)
    t0 = time.perf_counter()
    ids = MODEL.greedy_decode(src.to(DEVICE), max_len=CK["max_tgt"], banned=banned)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "t": ev["t"], "spivak_class": ev["spivak_class"],
        "confidence": ev["confidence"], "input": src_text,
        "commentary": restore_names(VOCAB.decode(ids[0].tolist()), {}),
        "ms": round(ms, 1),
    }


SAMPLING_TEMPERATURE = 0.9
SAMPLING_TOP_P = 0.9


@torch.no_grad()
def variation_for(idx, seed=None):
    """OPTIONAL stochastic decoding, for presentation only.

    Uses torch.multinomial over the temperature-scaled, top-p-filtered distribution
    instead of argmax. The grounding mask is applied before the softmax exactly as in
    the greedy path, so illegal player/team tokens have probability 0 and cannot be
    sampled. Does NOT replace the automatic greedy commentary.
    """
    ev = EVENTS[idx]
    src_text = build_input(ev, HOME, AWAY)
    ex = {"input": src_text, "target": "", "grounding": {}}
    ds = CommentaryDataset([ex], VOCAB)
    src, _ = collate([ds[0]])
    src = src.to(DEVICE)
    banned = build_banned_mask([ex], VOCAB, DEVICE)
    if seed is None:
        seed = random.randrange(2 ** 31 - 1)
    t0 = time.perf_counter()
    ids = sample_decode(MODEL, src, banned, CK["max_tgt"],
                        SAMPLING_TEMPERATURE, SAMPLING_TOP_P, seed)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "t": ev["t"], "spivak_class": ev["spivak_class"],
        "commentary": restore_names(VOCAB.decode(ids[0].tolist()), {}),
        "ms": round(ms, 1), "seed": seed,
        "temperature": SAMPLING_TEMPERATURE, "top_p": SAMPLING_TOP_P,
    }


def _warmup():
    """First CUDA decode costs ~2.6 s of kernel warmup. Pay it at startup so the
    live generation during the demo is fast."""
    if EVENTS:
        t0 = time.perf_counter()
        generate_for(0)
        print(f"warmup decode: {(time.perf_counter()-t0)*1000:.0f} ms "
              f"(subsequent calls are much faster)")


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Football Commentary Generator</title><style>
body{font-family:system-ui,Segoe UI,Arial;margin:0;padding:18px;background:#12151a;
color:#e8eaed}
.wrap{max-width:1250px;margin:0 auto}
h1{font-size:19px;margin:0 0 2px}.sub{color:#9aa0a6;font-size:12px;margin-bottom:14px}
.grid{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.left{flex:1 1 620px;min-width:340px}
video{width:100%;background:#000;border-radius:8px;display:block}
.right{flex:1 1 380px;min-width:320px;background:#1a1d23;border:1px solid #3c4043;
border-radius:8px;padding:12px;max-height:70vh;overflow-y:auto}
.k{color:#9aa0a6;font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.ev{border-left:3px solid #f9ab00;padding:8px 10px;margin:10px 0;background:#20242b;
border-radius:0 6px 6px 0}
.ev.goal{border-left-color:#81c995}
.hd{font-size:12px;color:#f9ab00;margin-bottom:4px}
.hd.goal{color:#81c995}
.tx{font-size:14px;color:#e8eaed}
.mono{font-family:Consolas,monospace;font-size:11px;color:#8ab4f8;margin-top:5px;
word-break:break-word}
.ms{font-size:11px;color:#9aa0a6;margin-top:4px}
.var{margin-top:8px;padding:7px 9px;background:#1b2330;border-left:2px solid #8ab4f8;
border-radius:0 4px 4px 0}
.vh{font-size:10px;color:#8ab4f8;text-transform:uppercase;letter-spacing:.5px;
margin-bottom:3px}
.vb{background:transparent;border:1px solid #3c4043;color:#9aa0a6;font-size:11px;
padding:4px 8px;border-radius:4px;cursor:pointer;margin-top:7px}
.vb:hover{border-color:#8ab4f8;color:#8ab4f8}
button{background:#1a73e8;color:#fff;border:none;padding:8px 14px;border-radius:6px;
cursor:pointer;font-size:14px;margin-right:6px}
.up{margin-top:10px}
.up b{color:#f9ab00}
.note{margin-top:14px;padding:10px 12px;background:#2a2118;border-left:3px solid
#f9ab00;border-radius:4px;font-size:12px;color:#fdd663}
</style></head><body><div class="wrap">
<h1>Football Commentary Generator</h1>
<div class="sub">Manchester United vs Arsenal &middot; official SoccerNet TEST split
&middot; commentary generated live by the Transformer as the clip reaches each
detected event</div>
<div class="grid">
 <div class="left">
  <video id="v" controls preload="metadata" src="/video"></video>
  <div class="up">
   <button onclick="jump(0)">Jump to first event</button>
   <button onclick="jumpGoal()">Jump to the GOAL</button>
   <button onclick="reset()">Reset feed</button>
  </div>
  <div class="up mono" id="next"></div>
  <div class="note"><b>What is live:</b> action spotting ran once over the half at
  startup (it is a batch model). Every commentary line below is generated by the
  Transformer at the moment the video reaches that event &mdash; the millisecond
  timing shown is real server-side greedy decoding.</div>
 </div>
 <div class="right">
  <div class="k">Live commentary feed</div>
  <div id="feed"></div>
 </div>
</div></div>
<script>
let EV=[], done={};
function esc(s){var d=document.createElement('div');d.innerText=s;return d.innerHTML}
function fmt(t){return Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0')}
fetch('/api/events').then(r=>r.json()).then(d=>{EV=d;upNext();});
const v=document.getElementById('v');
function upNext(){
  const t=v.currentTime||0;
  const n=EV.find(e=>e.t>t);
  document.getElementById('next').innerText =
    EV.length? (n? 'next detection: '+fmt(n.t)+'  '+n.spivak_class+
      '  (conf '+n.confidence.toFixed(2)+')' : 'no further detections') : '';
}
function fire(i){
  if(done[i])return; done[i]=1;
  fetch('/api/generate?idx='+i).then(r=>r.json()).then(d=>{
    const goal = d.spivak_class==='Goal';
    const el=document.createElement('div');
    el.className='ev'+(goal?' goal':'');
    el.innerHTML='<div class="hd'+(goal?' goal':'')+'">'+fmt(d.t)+'  '+
      esc(d.spivak_class)+'  &middot; detector confidence '+d.confidence.toFixed(2)+
      '</div><div class="tx">'+esc(d.commentary)+'</div>'+
      '<div class="mono">'+esc(d.input)+'</div>'+
      '<div class="ms">greedy &middot; generated live in '+d.ms+' ms</div>'+
      '<button class="vb" onclick="vary('+i+',this)">Generate Variation</button>'+
      '<div class="vars"></div>';
    document.getElementById('feed').appendChild(el);
    el.scrollIntoView({behavior:'smooth',block:'end'});
  });
}
v.addEventListener('timeupdate',()=>{
  const t=v.currentTime;
  EV.forEach((e,i)=>{ if(t>=e.t && t<e.t+8) fire(i); });
  upNext();
});
function vary(i,btn){
  btn.disabled=true; btn.innerText='Sampling...';
  fetch('/api/variation?idx='+i).then(r=>r.json()).then(d=>{
    const box=btn.parentElement.querySelector('.vars');
    const el=document.createElement('div');
    el.className='var';
    el.innerHTML='<div class="vh">Variation &middot; sampling T='+d.temperature+
      ', top-p='+d.top_p+' &middot; seed '+d.seed+'</div>'+
      '<div class="tx">'+esc(d.commentary)+'</div>'+
      '<div class="ms">sampled live in '+d.ms+' ms</div>';
    box.appendChild(el);
    btn.disabled=false; btn.innerText='Generate Variation';
  }).catch(e=>{ btn.disabled=false; btn.innerText='Generate Variation'; });
}
function jump(i){ if(EV[i]) v.currentTime=Math.max(0,EV[i].t-6), v.play(); }
function jumpGoal(){ const i=EV.findIndex(e=>e.spivak_class==='Goal');
  if(i>=0){ v.currentTime=Math.max(0,EV[i].t-6); v.play(); } }
function reset(){ done={}; document.getElementById('feed').innerHTML=''; }
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _video(self):
        """Serve the mp4 with HTTP Range support - without it the browser cannot
        seek a 160 MB file, and playback often fails outright."""
        size = VIDEO.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(VIDEO, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    return          # browser seeked away; normal
                remaining -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif u.path == "/video":
                self._video()
            elif u.path == "/api/events":
                self._send(200, json.dumps(EVENTS), "application/json")
            elif u.path == "/api/variation":
                idx = int(parse_qs(u.query).get("idx", ["0"])[0])
                if not 0 <= idx < len(EVENTS):
                    raise ValueError("index out of range")
                self._send(200, json.dumps(variation_for(idx), ensure_ascii=False),
                           "application/json; charset=utf-8")
            elif u.path == "/api/generate":
                idx = int(parse_qs(u.query).get("idx", ["0"])[0])
                if not 0 <= idx < len(EVENTS):
                    raise ValueError("index out of range")
                self._send(200, json.dumps(generate_for(idx), ensure_ascii=False),
                           "application/json; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")
        except Exception as exc:
            try:
                self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                           "application/json")
            except Exception:
                pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    _warmup()
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Video demo ready at {url}   (Ctrl+C to stop)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")

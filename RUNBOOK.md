# RUNBOOK — presentation day

Everything runs **fully offline**. All weights, features and video are local.

```powershell
cd "C:\Users\Asus\OneDrive\Desktop\NLP Project"
```

**Two models exist. Know which one you are showing.**

| | V1 | V2 |
|---|---|---|
| checkpoint | `checkpoints/full_best.pt` | `checkpoints/v2_best.pt` |
| event classes | 15 (62% lumped as `<no_event>`) | 23 (`<no_event>` split into 8 actions) |
| BLEU-4 (7,826 official test) | 13.25 | **21.43** |
| val loss | 0.4097 | **0.3732** |
| used by | `demo_json.py`, `demo_live.py`, `demo_app.py` | `demo_final.py`, `timeline.py` |

---

## 1. THE MAIN DEMO — video + live commentary  ★★

```powershell
python demo_final.py
```

http://127.0.0.1:8020. Wait for **"Demo ready"** (~30 s: model load + warm-up).

Match video plays with **YOLO person boxes** overlaid. As the playhead reaches each
event, **V2 generates that line at runtime** and it appears in the feed with the real
decode time in ms.

Controls: **Jump to the GOAL** · **First event** · **Reset feed** · toggles for YOLO
boxes and auto-generate. Every card has **Generate Variation** (sampling, T=1.0) and
**Show JSON sent to the model**.

### The one thing you must say out loud

The feed mixes **two different sources**, colour-coded in the UI:

- **VIDEO-DERIVED (orange), 6 events.** Detected from the video by a pretrained
  action-spotting model (Yahoo *spivak*, CC-BY-4.0) over this clip's frame features.
  5 corners + the goal at **29:20, confidence 0.90**. These carry **no player
  identity** — the model gets anonymous `<PLAYER_k>` slots and the output keeps the
  placeholders instead of inventing names.
- **ANNOTATION-DERIVED (blue), 38 events.** SoccerNet's human commentary annotations,
  re-labelled into fine-grained actions (pass / shot / cross / foul / save / dribble /
  offside / clearance). These carry grounded players, so they **do** have names.
  **They were NOT detected from the video.**

> *"Only the orange events came from the pixels. The blue ones come from SoccerNet's
> annotations — my contribution there is the Transformer that turns them into
> commentary, and the re-labelling that made it possible."*

**Show the JSON** on the goal card — it traces detection → canonical event →
encoder input → raw tagged output → decoding parameters. That is the whole pipeline
on one screen.

---

## 2. THE RICH TIMELINE — 38 events, one half

```powershell
python timeline.py --variation
```

Prints every annotated event of the first half with V2 commentary generated live,
plus a sampled alternative per line. Writes `outputs/timeline_half1_var.json`.

**Use `--variation`.** Greedy alone produces only **13 distinct templates across 38
events** (11 fouls read near-identically). With sampling that becomes **86**.

---

## 3. JSON → COMMENTARY (V1) — best for explaining the architecture

```powershell
python demo_json.py
```

http://127.0.0.1:8010. Type a JSON event, press Generate. Shows the exact
`structured_input` and the grounding map, so you can trace *your JSON → structured
representation → Transformer → tagged output → names substituted back*.

`minute` is minutes **within a half** (max 54 in training), not absolute match time.
A 90th-minute event is `half: 2, minute: 45`.

## 4. EVENT BROWSER (V1) — pick any of 7,826 test events

```powershell
python demo_live.py
```
http://127.0.0.1:8000.

## 5. OFFLINE FALLBACK — if anything breaks

```powershell
start outputs\demo_sidebyside.mp4
```
30 s, plays in any player. No Python, no GPU, no internet.

**Ports: 8000 / 8010 / 8020 — one server per port at a time.**

---

## If asked to prove the anti-hallucination claim

```powershell
python test_grounding.py --ckpt checkpoints/full_best.pt --data data/processed.json --n 200
```

Prints 5 checks and `GROUNDING GUARANTEE: VERIFIED`. The one to point at is the
**adversarial** test: every banned player tag is forced to a **+1e9** logit — the
model maximally "wants" an ungrounded player — and it still cannot emit one, because
the mask sets those logits to `-inf` before the argmax. Structural, not learned.

## Numbers

| Metric | V1 | V2 |
|---|---|---|
| BLEU-4, 7,826 official test | 13.25 | **21.43** |
| Val loss / perplexity | 0.4097 / 1.51 | **0.3732 / 1.45** |
| Ungrounded slots, delivered | 0.00% | **0.00%** |
| Ungrounded, unconstrained | 0.115% | **0.026%** |
| Trainable parameters | 8,491,651 | 8,489,599 |
| Player-hash resolution when parsing | 39,097 / 39,097 = 100% | same |
| Spivak spotting, published avg-mAP | — | 0.714 |

**If asked why BLEU nearly doubled — answer precisely:**
V2's per-n-gram precisions are actually slightly *lower* (p1 46.2 vs 50.9). The gain
is the **brevity penalty**: V1's was **0.57** (it generated ~36% too few tokens
because `<no_event>` collapsed to short stubs), V2's is **1.00**. The fix was
systematic under-generation, not n-gram accuracy. Saying "V2's n-grams are more
accurate" is wrong and checkable.

---

## Known limitations — say these before you are asked

1. **Player identity cannot come from video.** No jersey OCR; tracking is too
   fragmented (median 1.58 s, 544 tracks over 3 min). Video-derived lines use
   `<PLAYER_k>` placeholders.
2. **Slot count for video events is assumed** from the event type (goal→2,
   corner→1), not detected.
3. **Goal roles can be reversed.** For the 29:51 annotation the model said Herrera
   passed to Young; in reality Herrera scored and Young assisted. There is no role
   marking in the input. Do not build the pitch around the goal sentence.
4. **Scorelines are invented** — the score is not in the input. A *content*
   hallucination, distinct from the *identity* one the system provably prevents.
5. **Greedy repeats within a class.** Use `--variation` / the Variation button.
6. **Free-kick execution is labelled `foul`** (the `free kick` pattern lives in the
   foul rule family). A known, accepted trade-off from the V2 re-labelling.
7. **`clearance` is thin** (268 training examples) and `offside` is templated
   (215 distinct word types across 941 examples) — expect repetitive offside text.

## What NOT to claim

- YOLO does **not** identify players. It detects *people*, including officials and
  crowd.
- The system does **not** work on arbitrary uploaded video (SoccerNet's precomputed
  features only exist for their matches).
- The spivak spotter is **Yahoo's pretrained model**, not yours. The Transformer,
  the grounding guarantee and the V2 re-labelling are yours.
- The fine-grained pass/shot/foul stream is **annotation-derived**, not detected.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Port busy | One server per port; close the other terminal |
| Video won't play or seek | `demo_final.py` re-remuxes `outputs/web/half1.mp4` via ffmpeg on start |
| First line slow | Warm-up runs at startup; wait for "Demo ready" |
| YOLO boxes missing | `python precompute_yolo.py` (~25 min, writes `outputs/yolo_boxes_half1.json`) |
| Variation looks identical | Click again — it retries, but the distribution is peaked |
| Anything at all fails | `start outputs\demo_sidebyside.mp4` |

**Do not run `train.py`** — it overwrites a checkpoint.
**Do not regenerate `data/vocab.json`** — V1's checkpoint depends on it for the
team-name mask (V2's checkpoint is self-contained and unaffected).

## Where things live

```
checkpoints/full_best.pt              V1 (BLEU 13.25)
checkpoints/v2_best.pt                V2 (BLEU 21.43)  <- used by demo_final
data/vocab.json / vocab_v2.json       must stay beside their checkpoints
data/processed.json / _v2.json        36,892 parsed examples
models/spivak/                        pretrained action spotting (Yahoo, CC-BY-4.0)
outputs/yolo_boxes_half1.json         precomputed YOLO overlay
outputs/spivak_detections_half1.json  video-derived events
outputs/timeline_half1_var.json       38-event rich timeline
outputs/web/half1.mp4                 browser-playable video
outputs/demo_sidebyside.mp4           offline fallback
web/final_demo.html                   main demo UI
CLAUDE.md                             full project record and decisions
SETUP.md                              what runs from a fresh clone
```

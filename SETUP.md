# SETUP — running this after cloning

## TL;DR

**The Transformer demos work immediately after cloning.** The video demos do not —
they need SoccerNet broadcast video, which is NDA-covered and cannot be redistributed.
You must download it with your own SoccerNet NDA password.

```powershell
git clone <repo-url>
cd "NLP Project"
pip install -r requirements.txt
python demo_json.py          # works right away -> http://127.0.0.1:8010
```

---

## What ships in the repo

| Included | Why |
|---|---|
| `checkpoints/full_best.pt` (35 MB) | the trained Transformer — so you never have to retrain |
| `data/vocab.json` | **must stay next to the checkpoint** (see warning below) |
| `data/processed.json` (28 MB) | 36,892 parsed examples, official train/valid/test splits |
| `data/splits.json` | official SoccerNet game-level split manifest |
| `outputs/test_full_eval.json` | full test-split evaluation (BLEU-4 = 13.25) |
| `outputs/spivak_detections_half1.json` | cached action-spotting detections |
| all `.py`, `web/json_demo.html`, `CLAUDE.md`, `RUNBOOK.md` | code + docs |

## What does NOT ship, and why

| Excluded | Reason | How to get it |
|---|---|---|
| `data/caption-2024/` | labels are public, but the folder also holds **NDA video + features** | `python download_and_parse.py download` (labels, ~8 min, no NDA) |
| `outputs/web/half1.mp4`, `*.mkv` | **NDA-covered broadcast video** — redistributing it violates the SoccerNet agreement | `python download_video.py` with your own NDA password |
| `*.npy` (ResNET features) | NDA-covered, derived from the video | see below |
| `models/spivak/` (75 MB) | third-party model (Yahoo, CC-BY-4.0) — legal to share but large | curl commands below |
| `yolov8n.pt` | auto-downloads on first use | nothing to do (needs internet once) |
| other epoch checkpoints | only the best one is needed | — |

---

## Works immediately after clone (no NDA, no downloads)

```powershell
python demo_json.py       # JSON -> structured input -> Transformer -> commentary
python demo_live.py       # browse any of 7,826 official test events
python generate.py --ckpt checkpoints/full_best.pt --data data/processed.json --split test --n 8
python test_grounding.py --ckpt checkpoints/full_best.pt --data data/processed.json --n 200
python vocab.py test --data data/processed.json
python model.py           # architecture self-test
```

These cover the Transformer, the grounding guarantee, and the evaluation numbers —
i.e. everything the project actually claims as its own contribution.

## Needs the SoccerNet NDA (video demos)

`demo_app.py`, `demo_render.py` and `spot_to_commentary.py` need media we cannot ship.

1. Get an NDA password from <https://www.soccer-net.org/data>.
2. Download the video (first half of the demo match):
   ```powershell
   $s = Read-Host "SoccerNet NDA password" -AsSecureString
   $env:SOCCERNET_PASSWORD = [System.Net.NetworkCredential]::new("", $s).Password
   python download_video.py
   Remove-Item Env:\SOCCERNET_PASSWORD
   ```
   (`Read-Host` keeps the password out of PowerShell history.)
3. Download the ResNET features (public, no NDA needed):
   ```powershell
   $g = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"
   $u = "https://exrcsdrive.kaust.edu.sa/public.php/webdav/$g/1_ResNET_TF2.npy" -replace ' ','%20'
   curl.exe -sL -u "9eRjic29XTk0gS9:SoccerNet" -o "data/caption-2024/$g/1_ResNET_TF2.npy" $u
   ```
4. Re-fetch the spivak action-spotting model (CC-BY-4.0):
   ```powershell
   $B = "https://huggingface.co/yahoo-inc/spivak-action-spotting-soccernet/resolve/main"
   $M = "models/spotting_test_resnet_normalized_confidence_zoo_lr1e-3_dwd2e-4_sr0.5_mu2.0/best_model"
   New-Item -ItemType Directory -Force "models/spivak/$M/variables" | Out-Null
   curl.exe -sL -o "models/spivak/$M/saved_model.pb"    "$B/$M/saved_model.pb"
   curl.exe -sL -o "models/spivak/$M/keras_metadata.pb" "$B/$M/keras_metadata.pb"
   curl.exe -sL -o "models/spivak/$M/variables/variables.index" "$B/$M/variables/variables.index"
   curl.exe -sL -o "models/spivak/$M/variables/variables.data-00000-of-00001" "$B/$M/variables/variables.data-00000-of-00001"
   curl.exe -sL -o "models/spivak/models/resnet_normalizer.pkl" "$B/models/resnet_normalizer.pkl"
   ```
   Requires TensorFlow (`pip install tensorflow`). Note: `huggingface_hub` fails with an
   SSL certificate error in some environments — curl works.
5. `demo_app.py` remuxes the `.mkv` to browser-playable `.mp4` automatically (needs `ffmpeg` on PATH).

---

## ⚠️ Two things that will silently break it

**1. Never regenerate `data/vocab.json`.** The checkpoint does not carry
`vocab_target_ids`, so `generate.load_model()` recovers them from `data/vocab.json`,
guarded by a byte-for-byte `itos` equality check. If the vocab is rebuilt with a
different token order the guard fails, prints a *warning only*, and decoding silently
continues **without the team-name grounding mask**. Keep the committed vocab.

**2. Never run `train.py`.** It overwrites `checkpoints/full_best.pt`. The committed
checkpoint is the one all reported numbers (BLEU-4 13.25, val loss 0.4097) refer to.

## Requirements

Python 3.13, PyTorch 2.9 (CUDA optional — CPU works, just slower). `tensorflow` and
`ultralytics` only for the video demos. See `requirements.txt`.

## Where to look next

- `RUNBOOK.md` — how to actually run the demos, what to say, troubleshooting
- `CLAUDE.md` — full project record: verified dataset schema, design decisions,
  measured results, and known limitations

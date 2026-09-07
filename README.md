# Football Commentary Generator

A custom PyTorch encoder–decoder **Transformer with multi-head attention** that
generates natural-language football commentary from structured match events.

```
SoccerNet event  →  grounded structured input  →  Transformer encoder
                 →  masked decoder + cross-attention  →  greedy decoding  →  commentary
```

## Results

| Metric | Value |
|---|---|
| BLEU-4 (official SoccerNet test split, 7,826 examples) | **13.25** |
| Validation loss / perplexity | 0.4097 / 1.51 |
| Trainable parameters | 8,491,651 |
| Ungrounded player references in output | **0.00%** |
| Player-entity resolution during parsing | 39,097 / 39,097 (100%) |

Trained on the **official** SoccerNet `caption-2024` train split (281 games,
21,672 examples), validated on the official valid split, and evaluated on the
official test split, which was never seen during development.

## Architecture

`d_model=256`, `nhead=8`, 4 encoder + 4 decoder layers, `dim_feedforward=1024`,
sinusoidal positional encoding, teacher forcing, AdamW, greedy autoregressive
decoding. All three attention mechanisms are present and verified: encoder
self-attention, masked decoder self-attention, and decoder cross-attention.

## Grounding: the model cannot invent a player

Player identity is enforced **structurally, not learned**:

1. Real names never enter the target vocabulary. The model emits positional tags
   (`<PLAYER_1>`, `<PLAYER_2>`), which are substituted with real names *after*
   generation. No parameter setting can produce a name.
2. Entity tags the input never established are masked to `-inf` before the
   `argmax`, so an ungrounded reference is unreachable rather than merely unlikely.

Verified adversarially in `test_grounding.py`: every banned tag is forced to a
`+1e9` logit — the model maximally "wants" to emit an ungrounded player — and
**zero** are ever selected.

## Quick start

```bash
git clone https://github.com/Vishwa-Thangapandiyan/football-commentary-generator.git
cd football-commentary-generator
pip install -r requirements.txt
python demo_json.py          # → http://127.0.0.1:8010
```

The trained checkpoint is included, so no retraining is needed.

## Demos

| Command | What it does |
|---|---|
| `python demo_json.py` | Type a JSON event → see the structured encoder input → generated commentary |
| `python demo_live.py` | Browse and generate for any of 7,826 official test events |
| `python demo_app.py` | Match video with commentary generated live at each detected event¹ |
| `python spot_to_commentary.py` | Full pipeline: video features → action spotting → Transformer¹ |
| `python test_grounding.py` | Adversarial proof of the grounding guarantee |

¹ Requires SoccerNet broadcast video, which is NDA-covered and **not** included.
See [SETUP.md](SETUP.md).

## Video → commentary pipeline

An optional secondary pipeline detects events from video without any manual
annotation, using a pretrained action-spotting model
([Yahoo spivak](https://huggingface.co/yahoo-inc/spivak-action-spotting-soccernet),
CC-BY-4.0) over SoccerNet ResNET features, then feeds detected events into this
Transformer. It correctly located the match's only goal at 29:20 with 0.90
confidence.

**Scope note:** action spotting supplies the event type and timestamp; player
identity is *not* recoverable from video without jersey OCR, so pipeline-generated
commentary contains no player names. This is stated rather than hidden.

## Known limitations

- Commentary generated from video has no player names (see above).
- Repeated events of the same type can produce identical text; optional
  `torch.multinomial` sampling (T=0.9, top-p=0.9) provides variation.
- Scorelines are not part of the input, so any scoreline in the output is
  ungrounded — a *content* hallucination, distinct from the *identity*
  hallucination the system provably prevents.
- Unlabelled "colour commentary" events carry little conditioning signal and
  mode-collapse; they are excluded from the video demo.

Full details, verified dataset schema, and every design decision are recorded in
[CLAUDE.md](CLAUDE.md). Presentation instructions are in [RUNBOOK.md](RUNBOOK.md).

## Data

[SoccerNet](https://www.soccer-net.org/) `caption-2024`. Caption labels are public;
broadcast video and derived features require the SoccerNet NDA and are excluded
from this repository.

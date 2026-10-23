# Football Commentary Generator: Transformer vs Dragon Hatchling (BDH)

Two neural architectures, trained from scratch on SoccerNet, that turn a structured
football event into a sentence of live commentary, with a structural guarantee
that **no player is ever invented**.

```
SoccerNet event + independent action annotations + lineup
        │
        ├──►  encoder–decoder Transformer   (multi-head self-, masked- and cross-attention)
        └──►  BDH, "Baby Dragon Hatchling"  (Pathway, 2025: sparse positive neurons, linear attention)
                     │
        greedy decoding with grounded constrained decoding (shared by both)
                     │
        "goal! Ander Herrera finds some space inside the box to get on the end of a
         pass from Ashley Young, then sends the ball low into the middle of the net."
```

This repository compares the two architectures **like for like**: same data, same
official splits, same vocabulary, same decoding, same grounding mask, same metrics,
three training seeds each.

## Headline results

Official SoccerNet-Caption **test split** (98 games, 7,826 comments), mean ± sd over
3 seeds. Full tables, per-seed numbers and significance tests: **[RESULTS.md](RESULTS.md)**.

| System | Params | BLEU-4 | METEOR | ROUGE-L | CIDEr | Invented players |
|---|---|---|---|---|---|---|
| Best lookup baseline (same input) | – | 15.33 | 27.41 | 36.74 | 56.11 | – |
| Transformer | 8.50M | 18.67 ± 0.89 | 29.59 ± 0.50 | 40.56 ± 0.74 | 65.26 ± 1.53 | **0.00%** |
| BDH | 7.42M | **18.82 ± 0.22** | **29.80 ± 0.28** | **40.83 ± 0.48** | **70.69 ± 1.59** | **0.00%** |

- **Both architectures clearly beat retrieval of the most frequent sentence**
  (+3.3–3.8 BLEU, p < 0.001).
- **On BLEU, METEOR and ROUGE-L they tie.** Seed-to-seed variance is larger than
  the architecture difference: a single-run significance test picks a different
  "winner" depending on the seed.
- **BDH writes more informative and more varied commentary.** Its CIDEr is higher
  on every seed, with 1.8× more distinct bigrams, using 13% fewer parameters.
- **BDH learns faster early** (validation loss 0.85 vs 1.34 after one epoch), but
  the Transformer ends slightly lower (0.397 vs 0.409) and trains faster per epoch.

## What makes the comparison trustworthy

**1. No answer leakage in the input.** An earlier version (V2) labelled each event
type with keyword rules run over the *reference* commentary, so the model was partly
shown the answer. The final data (**V3**) instead attaches the independent human
**SoccerNet-v2 action annotations** (shot on target, foul, corner, ...) within a
time window chosen on the training split only. They agree with the commentary's
own event labels 85–87% of the time on held-out splits. See
[RESULTS.md §1](RESULTS.md#1-data-versions).

**2. Grounding is structural, not learned, for both architectures.**
- Real names never enter the vocabulary. Models emit `<PLAYER_1>`, `<PLAYER_2>`,
  and names are filled in afterwards from the match lineup.
- Any placeholder the input did not establish has its logit set to −∞ before every
  decoding step, so an ungrounded reference is unreachable.
- `test_grounding.py` proves this adversarially: banned tags forced to +1e9 logits
  are never selected.

**3. Honest baselines and limitations.** Lookup baselines with the same input,
multiple seeds, paired bootstrap tests, and an ablation without the event label.
Known weaknesses are measured rather than hidden: 81–94% of outputs copy a training
sentence verbatim (SoccerNet commentary is very formulaic), scorelines are
invented, and scorer/assister roles can be swapped.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Vishwa-Thangapandiyan/football-commentary-generator.git
cd football-commentary-generator
pip install -r requirements.txt

python compare.py            # both models, side by side, on 8 test events
python demo_json.py          # type an event as JSON → commentary  (http://127.0.0.1:8010)
```

Both trained V3 checkpoints are included, so nothing needs retraining. If you cloned
without `--recurse-submodules`, run `git submodule update --init` to fetch BDH.

## Using either model

Every script loads checkpoints through `generate.load_model`, which detects the
architecture, so **any script runs either model** by changing `--ckpt`:

| Checkpoint | Model | Data |
|---|---|---|
| `checkpoints/transformer_v3_best.pt` | Transformer, V3 (best of 3 seeds by validation loss) | `data/processed_v3.json` |
| `checkpoints/bdh_v3_best.pt` | BDH, V3 (best of 3 seeds by validation loss) | `data/processed_v3.json` |
| `checkpoints/full_best.pt` | Transformer, original V1 run (used by the demos) | `data/processed.json` |

```bash
python generate.py --ckpt checkpoints/bdh_v3_best.pt --data data/processed_v3.json --split test --n 8
python test_grounding.py --ckpt checkpoints/bdh_v3_best.pt --data data/processed_v3.json --n 200
python compare.py --full     # whole test split, both models (long on a laptop GPU)
python metrics.py "outputs/seeds/eval_*_v3_*.json" --train-data data/processed_v3.json
```

## Reproducing the study

```bash
# 1. data (public labels, no NDA)
python download_and_parse.py download && python download_and_parse.py parse
python build_v3.py download && python build_v3.py window && python build_v3.py build
python vocab.py build --data data/processed_v3.json --out data/vocab_v3.json

# 2. sanity gates: both models must memorise 32 examples
python train.py overfit --data data/processed_v3.json --vocab data/vocab_v3.json
python train_bdh.py overfit --data data/processed_v3.json --vocab data/vocab_v3.json

# 3. train + evaluate one seed locally ...
python train_bdh.py train --data data/processed_v3.json --vocab data/vocab_v3.json --tag bdh_v3 --seed 0
python train.py train --data data/processed_v3.json --vocab data/vocab_v3.json --tag tf_v3 --seed 0

# ... or all seeds of both models on a Colab GPU (~1 h on an A100)
colab exec -s <session> --timeout 9000 --env DSET=v3 -f colab_seeds.py
python colab_watch.py -s <session> --last eval_tf_v3_a100_s2   # pulls results, stops the VM

# 4. metrics and significance (CPU only)
python metrics.py "outputs/seeds/eval_*_v3_*.json" --train-data data/processed_v3.json
python metrics.py --compare outputs/seeds/eval_bdh_v3_a100_s0.json outputs/seeds/eval_tf_v3_a100_s0.json
```

## The two architectures

| | Transformer (`model.py`) | BDH (`bdh_model.py`) |
|---|---|---|
| Structure | encoder–decoder, 4 + 4 layers | decoder-only, 6 layers, input as a prefix |
| Width | d_model 256, 8 heads, FFN 1,024 | d 256, 4 heads, 8,192 sparse neurons per layer |
| Attention | softmax self-, masked self- and cross-attention | linear attention over sparse positive activations |
| Positions | fixed sinusoidal | rotary |
| Optimiser | AdamW, lr 3e-4, wd 0.01 | AdamW, lr 1e-3, wd 0.1 (upstream defaults) |
| Precision | fp32 | bf16, with attention forced to fp32 (fp16 overflows) |

BDH is the official implementation from
[pathwaycom/bdh](https://github.com/pathwaycom/bdh) (MIT), included unmodified as a
submodule; `bdh_model.py` only adapts it to prefix-LM data-to-text. The loss is
taken on target tokens only, so validation losses are directly comparable with the
Transformer's.

## Optional: video → commentary

A secondary pipeline detects events from broadcast video with a pretrained
action-spotting model
([Yahoo spivak](https://huggingface.co/yahoo-inc/spivak-action-spotting-soccernet),
CC-BY-4.0), which predicts the same 17 classes as the SoccerNet-v2 annotations V3
uses, and feeds them to the commentary model. It found the demo match's only goal
at 29:20 with 0.90 confidence. Player identity is **not** recoverable from video, so
video-derived commentary has no names. YOLO boxes in the demos locate people only.

| Command | What it does |
|---|---|
| `python demo_json.py` | JSON event → structured input → commentary |
| `python demo_live.py` | browse and generate for any official test event |
| `python demo_final.py` | match video + video-detected and annotated events, narrated live¹ |
| `python spot_to_commentary.py` | video features → action spotting → commentary¹ |

¹ Needs SoccerNet broadcast video, which is NDA-covered and **not** included; see
[SETUP.md](SETUP.md).

## Repository layout

```
download_and_parse.py   SoccerNet labels → grounded (input, target) pairs   (V1)
sublabel_v2.py          V2 keyword sub-labels (kept for history; leaks the reference)
build_v3.py             V3: join SoccerNet-v2 action annotations           (final data)
vocab.py                word-level vocabulary + target-side mask
model.py, train.py      encoder–decoder Transformer + trainer
bdh_model.py, train_bdh.py   BDH adapter + trainer/evaluator
generate.py             shared loader (both architectures), greedy decoding, BLEU, grounding audit
compare.py              BDH vs Transformer side by side / full split
metrics.py              BLEU, METEOR, ROUGE-L, CIDEr, diversity, copying, paired bootstrap
test_grounding.py       adversarial proof of the grounding guarantee
analyze_neurons.py      neuron-level concept selectivity, BDH vs Transformer
colab_seeds.py, colab_watch.py   multi-seed runs on Colab; auto-pull and auto-stop
eval_v2.py              V1 vs V2 evaluation (history)
demo_*.py, web/         demos
third_party/bdh/        official BDH code (git submodule)
outputs/seeds/          every generated test sentence of every run, plus histories
```

## Data and licences

[SoccerNet](https://www.soccer-net.org/) `caption-2024` commentary labels and
SoccerNet-v2 action labels are public. Broadcast video and features derived from
it require the SoccerNet NDA and are excluded from this repository, as are any
frames from it. BDH © Pathway Technology, MIT licence.

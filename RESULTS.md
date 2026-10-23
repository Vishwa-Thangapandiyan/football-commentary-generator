# Results

All numbers are on the **official SoccerNet-Caption test split** (98 games, 7,826
commentary examples), which was never used for training, model selection or the
choice of any data-processing parameter. Every number in this file can be
regenerated from files in the repository; the source of each table is given.

## 1. Data versions

The three versions differ **only in the encoder input**. Targets, splits, row order
and the player grounding are identical across all three.

| Version | Event information in the input | Valid for evaluation? |
|---|---|---|
| **V1** | SoccerNet-Caption `label` of the comment (59% of rows are empty → `<no_event>`) | yes |
| **V2** | V1, with `<no_event>` split into pass / shot / foul / ... by keyword rules over the **reference commentary** | **no: leaks the answer** |
| **V3** | V1 + up to 4 **SoccerNet-v2 action annotations** (`Labels-v2.json`) within −45 s / +20 s of the comment | yes |
| V3 (no `<EVT>`) | V3 with the commentary's own label removed (actions only) | yes, ablation |

**Why V2 is not used for conclusions.** V2's sub-labels were extracted from the very
sentence the model must produce, on every split including test, so the input
partly contains the answer. V2 numbers are reported in section 6 for history only.

**How V3 was built** (`build_v3.py`). SoccerNet-v2 action spotting labels are a
separate human annotation (17 classes, each with a timestamp and a team). For each
comment, actions of the same half are attached if they fall in a window around the
comment's timestamp. The window was chosen on **train only**: the narrowest window
reaching 95% of the best achievable agreement between the joined actions and the
commentary's own labelled events (corner ↔ Corner, goal ↔ Goal, card ↔ card,
substitution ↔ Substitution). It held up on data it was not chosen on:

| Split | Agreement (labelled comments whose window contains the matching action) | Comments with ≥1 action |
|---|---|---|
| train (window chosen here) | 85.9% | 88.0% |
| valid | 87.1% | 87.2% |
| test | 85.5% | 87.7% |

Source: `python build_v3.py window`, `data/v3_window.json`.

## 2. Main result: BDH vs Transformer on V3

Both models: 10 epochs, official train split, best epoch by validation loss, greedy
decoding with grounded constrained decoding, **3 seeds each**, trained on a Colab
A100. Mean ± standard deviation over seeds.

| System | Params | BLEU-4 | METEOR | ROUGE-L | CIDEr | distinct-2 | Output/ref length |
|---|---|---|---|---|---|---|---|
| Lookup: one constant sentence | – | 8.46 | 23.34 | 25.66 | 9.49 | 0.01 | 0.59 |
| Lookup: event label + #players | – | 15.36 | 26.75 | 35.02 | 51.68 | 0.27 | 0.89 |
| Lookup: event label + nearest action + #players | – | 15.33 | 27.41 | 36.74 | 56.11 | 1.15 | 0.84 |
| **Transformer** (enc-dec) | 8.50M | 18.67 ± 0.89 | 29.59 ± 0.50 | 40.56 ± 0.74 | 65.26 ± 1.53 | 0.90 ± 0.06 | 0.95 |
| **BDH** (Dragon Hatchling) | 7.42M | **18.82 ± 0.22** | **29.80 ± 0.28** | **40.83 ± 0.48** | **70.69 ± 1.59** | **1.62 ± 0.05** | 0.92 |

"Lookup" baselines return the most frequent training sentence for inputs with the
same key, i.e. they see the same information as the models but cannot generate.

Per seed:

| Seed | BDH BLEU / CIDEr | Transformer BLEU / CIDEr |
|---|---|---|
| 0 | 18.64 / 69.52 | 19.16 / 65.00 |
| 1 | 18.75 / 70.04 | 19.20 / 66.90 |
| 2 | 19.07 / 72.50 | 17.64 / 63.88 |

Source: `outputs/metrics_v3.json` (`python metrics.py ...`), raw outputs in
`outputs/seeds/eval_*_v3_*.json`.

### Findings

1. **Both models clearly beat lookups with the same information**: +3.3 to +3.8
   BLEU-4 (paired bootstrap, 1,000 resamples, p < 0.001) and +9 to +15 CIDEr.
2. **BLEU, METEOR and ROUGE-L: a tie.** Within a single seed the paired bootstrap
   declares a winner, but the winner changes with the seed: BDH −0.52 [−0.91, −0.11]
   (seed 0), −0.45 [−0.91, +0.03] (seed 1), +1.43 [+1.03, +1.87] (seed 2).
   **Training-seed variance is larger than the architecture difference**, so
   single-run comparisons of these models are not meaningful.
3. **BDH produces more informative and more diverse commentary.** CIDEr, which
   rewards rare, informative n-grams, is higher for BDH on **every** seed
   (70.7 vs 65.3), and BDH's outputs contain 1.8× more distinct bigrams.
4. **BDH is more stable across seeds** on V3 (BLEU sd 0.22 vs 0.89).
5. **The commentary's own event label still matters**: removing `<EVT>` costs
   1.5–2.2 BLEU and 6–9 CIDEr for both models, yet both still beat the
   label-based lookups using the SoccerNet-v2 actions alone.

| Ablation (seed 0) | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|
| BDH, V3 | 18.64 | 29.66 | 40.41 | 69.52 |
| BDH, V3 without `<EVT>` | 17.18 | 28.65 | 38.31 | 60.41 |
| Transformer, V3 | 19.16 | 29.54 | 40.97 | 65.00 |
| Transformer, V3 without `<EVT>` | 16.97 | 28.25 | 39.42 | 58.94 |

## 3. Training behaviour

| | Transformer | BDH |
|---|---|---|
| Trainable parameters | 8,501,911 | 7,417,344 |
| Validation loss after epoch 1 | 1.341 | 0.852 |
| Validation loss after epoch 5 | 0.480 | 0.431 |
| Best validation loss (mean of 3 seeds) | **0.397** | 0.409 |
| Seconds per epoch (A100) | **24** | 34 |

BDH learns much faster in the first epochs with 13% fewer parameters, but plateaus
earlier and ends with a slightly higher validation loss. It is also slower per
epoch: its sparse layer is 8,192 neurons wide.

Source: `outputs/seeds/history_*_v3_*.json`.

## 4. Grounding: no invented players, for either architecture

| | Transformer | BDH |
|---|---|---|
| Ungrounded player tags, shipped (constrained) output | **0.00%** (all seeds) | **0.00%** (all seeds) |
| Ungrounded player tags, unconstrained debug output | 0.08% | 0.16% |
| Real names in the output vocabulary | 0 | 0 |

Names never enter the vocabulary, and placeholder tags the input did not establish
are masked to −∞ before every decoding step. The mask is architecture-independent,
which is why the guarantee carries over to BDH unchanged. `test_grounding.py`
verifies it adversarially (banned tags forced to +1e9 logits are never selected).

## 5. Honest limitations

- **SoccerNet commentary is extremely formulaic.** 81–94% of generated sentences
  are verbatim copies of some training sentence (both models). The task is closer
  to choosing and filling the right template than to free composition.
- **Scorelines are invented.** The score is not in the input, so any score in the
  output is ungrounded (a content hallucination, distinct from the identity
  hallucination that is prevented).
- **Roles can be swapped.** Both players are grounded, but the model may make the
  assister the scorer; there is no role information in the input.
- **The comment's `<EVT>` label and player list come from the dataset annotation**
  of that comment (standard data-to-text content planning, as in WebNLG). A fully
  automatic system would have to predict them; the "no `<EVT>`" ablation measures
  part of that gap.
- **Neuron-level interpretability (preliminary, V2 data, BDH epoch 6):** at a
  matched neuron count, BDH's single neurons were not more selective for football
  concepts than the Transformer's feed-forward units (`analyze_neurons.py`,
  `outputs/neurons_v2_epoch06.json`). BDH's synapse-level claims were not tested.

## 6. History: V1 and V2 (V2 leaks the reference)

| Model | Data | Seeds | BLEU-4 | Val loss |
|---|---|---|---|---|
| Transformer (original run, RTX 4060) | V1 | 1 | 13.25 | 0.410 |
| Transformer (original run, RTX 4060) | V2 | 1 | 21.43 | 0.373 |
| Transformer | V2 | 3 | 20.58 ± 0.15 | 0.372 |
| BDH | V2 | 3 | 20.29 ± 0.82 | 0.383 |

V1's low BLEU is mostly a brevity penalty (0.57): with 59% of inputs reading
`<no_event>`, the model fell back on short, generic sentences. V2 removed that
problem by reading the event type off the reference, which is why its scores are
higher than V3's and cannot be compared fairly.

Sources: `outputs/test_full_eval.json`, `outputs/eval_v2_compare.json`,
`outputs/seeds/eval_*_v2_*.json`.

## Reproducing

```bash
python build_v3.py download && python build_v3.py window && python build_v3.py build
python vocab.py build --data data/processed_v3.json --out data/vocab_v3.json
# train + evaluate both models, 3 seeds (Colab A100, ~1 h):
colab exec -s <session> --timeout 9000 --env DSET=v3 -f colab_seeds.py
python metrics.py "outputs/seeds/eval_*_v3_*.json" --train-data data/processed_v3.json
python metrics.py --compare outputs/seeds/eval_bdh_v3_a100_s0.json outputs/seeds/eval_tf_v3_a100_s0.json
```

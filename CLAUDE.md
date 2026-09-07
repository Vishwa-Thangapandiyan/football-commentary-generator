# CLAUDE.md — Football Commentary Generator (Multi-Head Attention Transformer)

## 1. What this project is

Build a **generative** football commentary system:

```
SoccerNet event/context  ->  grounded structured input  ->  custom PyTorch
encoder-decoder Transformer  ->  autoregressive greedy decoding  ->  commentary
```

Academic project. The Transformer with **multi-head self-attention, masked decoder
self-attention, and cross-attention** is the deliverable. It must GENERATE, never
retrieve.

**Hard priorities, in order:**
1. Grammatical, football-specific output.
2. Output grounded in the given event/context.
3. **No hallucinated player identities.**
4. Demonstrably a real Transformer with multi-head attention.
5. Working end-to-end baseline BEFORE any extension.

**Governing rule: working baseline > ambitious broken system.**
One evening of implementation time. Presentation is the next day.

---

## 2. Verified environment (checked 2026-09-07, do not re-verify)

| Item | Value |
|---|---|
| Python | 3.13.15 |
| torch | 2.9.0+cu128, `cuda.is_available() == True` |
| GPU | RTX 4060 Laptop, 8188 MiB |
| SoccerNet | 0.1.62 (installed this session) |
| Also present | numpy 2.2.6, transformers 5.1.0, torchaudio, torchvision, opencv, tqdm |
| Working dir | `C:\Users\Asus\OneDrive\Desktop\NLP Project` |
| Git | **not** a git repository |

**Repo state at takeover: EMPTY.** The project brief referred to existing
`download_and_parse.py` and `render_demo.py` from earlier sessions. Neither exists
here or anywhere under `C:\Users\Asus` (searched). Everything is being written fresh.
Nothing was overwritten.

---

## 3. VERIFIED dataset schema — this section is ground truth

Source: official `SoccerNet` pip package, task `caption-2024`, file
`Labels-caption.json`. Two real files downloaded and inspected. **Every statement
below was read off real data, not assumed.** Do not contradict this section from
memory or from the original brief.

### 3.1 Access

- Game lists: `getListGames(split, task="caption")` -> train **281**, valid **92**,
  test **98** = **471** games. Resolves offline from the package.
- `Labels-caption.json` matches the generic `"Labels" in file` branch of
  `downloadGame`, served from public share `ZDeEfBzCzseRCLA`, password `"SoccerNet"`.
  **No NDA / no credentials needed.**
- **Do NOT use `downloadDataTask(task="caption-2024", ...)`** — it also pulls
  `1_baidu_soccer_embeddings.npy` and `2_baidu_soccer_embeddings.npy` per game
  (huge, useless to us tonight).
  Use instead:
  `downloader.downloadGames(files=["Labels-caption.json"], split=["train"], task="caption")`
  or per-game `downloadGame(game=..., files=["Labels-caption.json"], spl="train")`.
- **Size/cost:** ~75-95 KB per game, ~1 s each. All 471 games is about 35 MB, about
  10 minutes. Label data is effectively free — dataset size is NOT a real constraint.

### 3.2 GOTCHA — telemetry crash (verified, and verified fixed)

`SoccerNet.Downloader.downloadFile` fires a Google-Analytics call
(`report('UA-99166333-3', ...)`, Downloader.py:86) **after** the file is written.
In this environment that raises
`SSLError: CERTIFICATE_VERIFY_FAILED (ssl.google-analytics.com)` and propagates,
killing the loop even though the download itself succeeded.

**Required in every download script — verified working:**

```python
import SoccerNet.Downloader as DL
DL.report = lambda *a, **k: None      # no-op the analytics call
from SoccerNet.Downloader import SoccerNetDownloader, getListGames
```

Do not edit the installed package. Do not wrap in a bare `try/except` (it hides real
failures).

### 3.3 `Labels-caption.json` top-level keys (real)

```
timestamp, score, round, teams, lineup, referee, venue, attendance,
referee_matched, referee_found, coach_matched,
gameHomeTeam, home, gameAwayTeam, away, gameDate, annotations
```

- `gameHomeTeam` / `gameAwayTeam` -> e.g. `"Chelsea"` / `"Burnley"` (plain strings).
- `home` / `away` -> dicts: `{name, detail_link, hash, names[]}`.
- `score` -> `"1 - 1"`. `gameDate` -> `"21/02/2015 - 16:00"`.
- `lineup` -> `{home: {tactic, players[], coach[]}, away: {...}}`.
- `annotations` -> list, **78 entries** in the sample game.

### 3.4 Annotation entry — exact keys

```
important, gameTime, label, description, identified, anonymized, visibility, position
```

| Field | Real content |
|---|---|
| `gameTime` | `"2 - 49:59"`, i.e. `"<half> - MM:SS"`. Half is 1, 2, or **3** (added-time / pre-match block). |
| `position` | milliseconds within the half, as a **string**: `"2999000"`. |
| `label` | Event class **or empty string**. See 3.5. |
| `description` | Commentary with **real names**: `"Ben Mee (Burnley) seems to have picked up..."` |
| `identified` | Same text, entities replaced by **hash-tagged placeholders**: `"[PLAYER_SzUQvQ7r] ([TEAM_]) seems to have picked up..."` |
| `anonymized` | Fully de-identified: `"[PLAYER] ([TEAM]) seems to have..."` |
| `important` | bool. 17 True / 61 False in the sample. |
| `visibility` | `"shown"` for all 78 in the sample. |

### 3.5 `label` — CRITICAL correction to the original brief

The brief assumed `label` is a reliable event type. **It is empty about 59% of the
time.** Sample game distribution (78 annotations):

```
''(empty) 46 | corner 12 | substitution 4 | whistle 3 | injury 3 | y-card 3
time 2 | soccer-ball 2 | funfact 1 | r-card 1 | attendance 1
```

`soccer-ball` = **goal**. These are SoccerNet-v2 action classes.
Empty label = general colour commentary with no discrete event.

**Decision (see 4.2):** keep empty-label rows and map them to an explicit
`<no_event>` class. Do not silently drop 59% of the data, and do not invent a label.

### 3.6 Player grounding — the brief's plan is SUPERSEDED (in a good way)

The brief proposed: find real names -> replace with `<PLAYER_k>`. **Not needed.**
The dataset already ships the placeholder version.

- `identified` contains `[PLAYER_<hash>]`, e.g. `[PLAYER_SzUQvQ7r]`.
- The hash is a **stable entity key** matching `hash` in `lineup.<side>.players[]`.
- **Verified: 24/24 distinct player hashes in the sample game resolved to lineup
  entries. Zero misses.**

`lineup.<side>.players[]` entry — real keys:

```
hash, name, short_name, long_name, shirt_number, country,
captain, detail_link, facts[], lineup, starting
```

e.g. `{hash: "vBw5nir8", name: "Costa D.", long_name: "Diego Costa",
short_name: "Costa", shirt_number: "19", country: "Spain", lineup: 11,
starting: true}`. 18 players per side (11 starting + 7 subs); `lineup` is the
formation slot (1-11) or `null` for unused subs. `coach[]` has the same shape;
coach hashes appear in text as `[COACH_<hash>]`.

So the pipeline is:

```
identified  ->  [PLAYER_<hash>]  ->  <PLAYER_k>   (k = order of first mention)
                      |
                      +-> lineup[hash] -> real name + home/away side  (kept OUT of vocab)
```

`<PLAYER_k>` numbering is per-annotation, by order of first mention. Real names
**never enter the target vocabulary**, so the model structurally cannot hallucinate
a player identity. Restore names after generation from the saved mapping.

### 3.7 Team / referee / coach tags — KNOWN LIMITATION

- `[TEAM_]` appears **84 times in the sample and the hash slot is ALWAYS EMPTY.**
  Team identity is **not** recoverable from the tag.
- `[REFEREE]` (7x) carries no hash. `[COACH_<hash>]` does carry one.

**Consequence:** we cannot tell which of the two teams a given `[TEAM_]` refers to.
**Decision:** emit a single generic `<TEAM>` token in the target. Honest, matches the
data. Optional inference-time polish: substitute the team of `<PLAYER_1>` (known from
lineup side) — if used, label it a heuristic in the presentation.

### 3.8 Do NOT use `anonymized` as the training target

It collapses every player to the same `[PLAYER]` token, destroying the distinction
between, say, scorer and assister. `identified` is the only field that supports
per-player grounding. **The target is built from `identified`.**

### 3.9 Position/role info is MORE available than the brief assumed

`lineup` slot (1-11) + `tactic` (e.g. `"1-4-2-3-1"`) + `shirt_number` + `starting`
make role tags (`<KEEPER>` etc.) genuinely feasible. **Still Phase 7 — not tonight.**
Tonight stays with positional `<PLAYER_k>` tags exactly as the brief specified.

---

## 4. Decisions (made, with reasons — do not relitigate)

### 4.1 Scope

- Event source = the caption annotation's own `label`. **No `Labels-v2.json` join and
  no temporal alignment tonight.** (Phase 7.)
- Text/event -> commentary only. No video, no Baidu embeddings.

### 4.2 Empty labels

Map `""` -> `<no_event>`. Keeps about 59% of the data; the model learns
event-conditioned vs. general commentary. Fallback if generation quality is poor:
filter to labelled rows only (that would cut the sample game from 78 to 32 rows).

### 4.3 Input representation

Flat token sequence built ONLY from fields verified in section 3:

```
<EVT> soccer-ball <HALF> 2 <MIN> 35 <IMP> 1
<HOME> chelsea <AWAY> burnley
<P> <PLAYER_1> away <P> <PLAYER_2> away
```

Player slots are listed in the input **on purpose**. This is standard grounded
data-to-text (as in WebNLG): the content plan is given, the model produces the surface
realization. It is not leakage — it is the mechanism that makes identity hallucination
impossible. **State this explicitly in the presentation**, because it will be asked.

Team names ARE real strings in the input (from `gameHomeTeam` / `gameAwayTeam`), and
the target only ever contains generic `<TEAM>`. **Note:** that alone made team names
merely *unlikely* in output, not impossible — src and tgt share one vocabulary, so
"chelsea" was still emittable. The global target-vocabulary mask in **4.14** is what
actually makes team names unreachable. See 4.14.

### 4.4 Target representation

`identified` with `[PLAYER_<hash>]` -> `<PLAYER_k>`, `[COACH_<hash>]` -> `<COACH_k>`,
`[TEAM_]` -> `<TEAM>`, `[REFEREE]` -> `<REFEREE>`. Lowercased, simple regex word
tokenization with punctuation split off.

### 4.5 Tokenization

**Word-level.** No BPE / SentencePiece. Reasons: implementation speed, small corpus,
debuggability, easier to explain in the presentation. Min-frequency threshold ->
`<UNK>`. Specials: `<PAD> <BOS> <EOS> <UNK>` plus `<PLAYER_1..N>`, `<COACH_1..N>`,
`<TEAM>`, `<REFEREE>` and the input field markers.
**Vocab is saved to disk and reloaded at inference.**

### 4.6 Model

Custom PyTorch encoder-decoder using `nn.TransformerEncoder` / `nn.TransformerDecoder`
(real multi-head attention, not a pretrained generation API).
`d_model=256, nhead=8, enc_layers=4, dec_layers=4, ff=1024, dropout=0.1,
batch_first=True`.

### 4.7 Positional encoding

**Sinusoidal, fixed.** No extra parameters, no length-generalization surprises, and it
lets the actual sin/cos formula go on a slide. Documented choice.

### 4.8 Training

Teacher forcing, `CrossEntropyLoss(ignore_index=PAD)`, AdamW, gradient clipping.
Masks required: decoder causal mask, encoder padding mask, memory padding mask, target
padding mask. **No** AMP / LR schedule / accumulation unless a run proves it necessary.

### 4.9 Generation

**Greedy** autoregressive only tonight. No top-k / top-p / beam.

### 4.10 Data scope and splits (APPROVED — download all, split policy is binding)

Measured: **90 annotations per game** (78 and 102 in the two inspected games).

| Games | Examples |
|---|---|
| 20 (dev slice) | ~1,800 |
| 281 (official train) | ~25,000 |
| 471 (all splits) | ~42,000 |

**Download: all 471 games**, once, up front. ~90 KB and ~1 s per game, so ~35 MB and
~8 minutes total. Download cost and training cost are unrelated, so there is no
reason to ration the data.

**The official SoccerNet train / valid / test splits are BINDING.**
`getListGames(split, task="caption")` -> train 281, valid 92, test 98.
The manifest is written to `data/splits.json` at download time.

- **Never mix, merge, reshuffle or re-derive the splits.** No random re-splitting.
- **Dev/debug slice = first ~20 games, and it is for debugging only.** It is a subset
  of the official splits, not a replacement for them. Nothing trained on the dev slice
  is reported as a result.
- **Full run: train on the official train split (281 games), validate on the official
  valid split (92 games).**
- **The official test split (98 games) is reserved for final qualitative examples and
  metrics. Do not look at it during development, and never train on it.**

Rationale for the 20-game dev slice: bugs are found just as fast on a small slice, and
~1,800 examples against a 6-8k word-level vocab is far too thin to report — most words
would be seen once and collapse to `<UNK>`. The slice proves the pipeline; the official
train split produces the results.

### 4.12 Training budget (APPROVED — hard limits, not guidance)

The full training run is **capped at 10 epochs or 5 minutes per epoch, whichever comes
first**. We are explicitly NOT training to convergence.

- **Save a checkpoint after every completed epoch.** No exceptions — a killed run must
  never lose finished work.
- **If any single epoch exceeds 5 minutes: STOP the run.** Report the epoch timing and
  the current loss/checkpoint state, then wait for direction. Do not silently continue,
  do not shrink the model, do not reduce the data to fit the budget.
- Time each epoch explicitly and print the per-epoch wall time.

### 4.14 GROUNDED CONSTRAINED DECODING (APPROVED — structural, not learned)

**Requirement: generated entity placeholders are restricted to placeholders the
current input actually established.** If the input has only `<PLAYER_1>` and
`<PLAYER_2>`, generation must not produce `<PLAYER_3>` or `<COACH_1>`.

Constrained decoding turned out to be **trivial** in the existing greedy loop (one
`masked_fill` before the `argmax`), so it is implemented at the logit level rather
than as post-hoc filtering. That upgrades the property from "rare after training" to
"unreachable". Three layers:

1. **Per-example tag mask** (`build_banned_mask`) — entity tags not in this
   example's grounding get logit `-inf`. `<TEAM>` and `<REFEREE>` carry no
   per-example identity and are always permitted.
2. **Global target-vocabulary mask** — ids that never occur on the target side of
   the training data are also `-inf`. This closes a real hole: because src and tgt
   share one vocabulary, team names ("chelsea", "milan") were *emittable* even
   though no training target contains one. 162 input-only tokens are now masked.
   **The earlier claim in 4.3 that team names "cannot be hallucinated" was too
   strong — it was true only statistically until this mask existed.**
3. **Post-decode validation + fallback** (`enforce_grounding`) — defense in depth.
   Constrained decoding should make it a no-op; if a violation ever appeared, the
   tag is replaced with a safe generic phrase ("the player" / "the manager").
   **A raw placeholder tag or a broken sentence must never reach presented output,
   and an identity is NEVER restored for an ungrounded placeholder.**

Unconstrained output is still produced and stored as `generated_tagged_raw_debug`
**for debugging only**. It is never presented and never has names restored.

**Verified by `test_grounding.py` (all 5 checks PASS, `GROUNDING GUARANTEE:
VERIFIED`), against the stage-5 overfit checkpoint:**

| Check | Result |
|---|---|
| Full real names leaked into training targets | **0** (of 3,453 name-parts) |
| Complete team names emittable | **0 of 122** |
| Banned tags selected under **adversarial +1e9 logits** (50,975 banned pairs) | **0** |
| Allowed tags still selectable | yes |
| Ungrounded rate, UNCONSTRAINED (overfit ckpt) | 28.00% |
| Ungrounded rate, CONSTRAINED (overfit ckpt) | **0.00%** |
| Post-decode fallback rewrites needed | 0 |

**On the FULLY TRAINED model, official test split, 7,826 examples:**
unconstrained **0.12%** (9 examples), constrained **0.00%** (0).

Report this honestly: the trained model is already almost entirely grounded on its
own, so the constraint rarely binds. It is a guarantee that seldom needs to fire,
NOT a fix for a live 28% problem — the 28% figure came from the stage-5 overfit
checkpoint (trained on 32 examples) and must not be quoted as the model's behaviour.
The adversarial test is what makes it a guarantee regardless of how the model
happens to behave.

The adversarial test is the point: we do not check that the model *chose* not to
emit an ungrounded tag, we force it to maximally want one and confirm it still
cannot. That is why this is a guarantee and not a trained behaviour.

One honest caveat for the presentation: the single token `nice` remains emittable
because it is an ordinary commentary word that coincides with a club name. It
carries no identity in context, and no *complete* team name is producible.

### 4.13 Grounding is MEASURED, not asserted (`generate.py --bleu`)

Two distinct hallucination channels, tracked separately:

1. **Real name in the output** — structurally impossible: names never enter the
   target vocabulary, so no parameter can emit one. Verified in `vocab.py build()`
   (`leaked raw entity tags in vocab: 0`), not assumed.
2. **Ungrounded slot** — the model emits a `<PLAYER_k>` / `<COACH_k>` the input never
   established. This one IS possible, so `hallucination_audit()` measures its rate.

**Measured progression (all real runs, not estimates):**

| Model | BLEU-4 | Ungrounded slots (unconstrained) |
|---|---|---|
| Untrained (random init), 1,780 dev examples | 0.00 | 96.74% |
| Stage-5 overfit ckpt (32 examples), 300 dev examples | - | 28.00% |
| **Trained, 7,826 OFFICIAL TEST examples** | **13.25** | **0.12%** |

Delivered output is constrained, so the shipped ungrounded rate is **0.00%** in
every case above.

### 4.15 Observed output quality — honest limitations for the presentation

Read from real test-split output (`outputs/generated_test.json`). Do not oversell
these results; every item below is visible in the 8 sampled examples.

**What works.** Output is grammatical, football-specific, and uses the correct
grounded players. Event conditioning is real: `y-card` produces a booking, `corner`
a corner, `substitution` a substitution with the right two players in the right
roles ("X walks off the pitch to be replaced by Y").

**What does not.**
1. **Invented scorelines.** Example 1 ends "0: 2" — the score is NOT in the input,
   so any scoreline is ungrounded. This is a *content* hallucination, distinct from
   the *identity* hallucination that 4.14 eliminates. Fix: add score to the input
   representation (it exists in the schema as `score`), or ban digits at decode.
2. **Role confusion between grounded players.** Example 1 has Herrera lifting the
   ball over Ashley Young, who is his own teammate. The tags are grounded, but the
   model has no notion of which player is opponent vs. teammate. Both are marked
   `home` in the input, so the information is there and unused. Phase 7: role tags.
3. **Internal contradiction.** Example 7: "fails to score from the penalty spot ...
   are awarded a penalty kick!" — greedy decoding with no discourse planning.
4. **Lowercased sentence starts.** Cosmetic artifact of lowercasing in `tokenize`.
   A capitalization pass in `detokenize` would fix it; not done yet.
5. **Near-verbatim reproductions.** Example 5 matches its reference almost exactly.
   This is a TEST-split example the model never saw, and SoccerNet commentary is
   heavily templated (2,516 distinct word types across 963k training tokens), so
   this is legitimate — but expect the question and answer it with the type/token
   ratio, not with a claim of novelty.

### 4.16 Demo artifacts (built, verified)

- **`demo_live.py`** — `python demo_live.py` -> http://127.0.0.1:8000. Python stdlib
  only (no Flask/Streamlit/CDN), fully offline. Dropdown of 40 verified official
  test-split events + Generate. Proves the model generates live rather than replaying
  canned text. Bad input returns JSON errors the UI shows gracefully.
- **`demo_render.py --mode highlights`** -> **`outputs/demo_sidebyside.mp4`**
  (12.2 MB, 1280x620, 30 s). Video LEFT with YOLO boxes, generated commentary feed
  RIGHT accumulating at each event's real timestamp, encoder input below.
  **Two hard limits found while building it, both real:**
  1. `<no_event>` rows MODE-COLLAPSE. Four consecutive ones all generated
     "X is adjudged offside" while the references were a foul, a cleared cross, a
     misplaced pass and a failed pass. Their input carries almost no signal (half,
     minute, teams, one player), so the model falls back to one high-frequency
     template. Highlights mode therefore uses LABELLED events only.
  2. The annotation clock outruns the video: SoccerNet labels this half to 46:59
     but `1_224p.mkv` is exactly 45:00. The corner at 45:51 and the whistle at
     46:24 have NO footage. The renderer now drops such events loudly instead of
     silently writing empty spans (which had made it report 1125 frames while
     writing 750).
  Net: only 2 labelled events (corner 10:18, goal 29:51) are usable from half 1.
  Downloading `2_224p.mkv` would add 62 more annotations with better variety.
- **`demo_render.py --mode window`** -> continuous span (includes `<no_event>`;
  repetitive, kept only for debugging).
- **`outputs/demo_goal.mp4`** (earlier stacked layout, 18 MB) retained as a fallback.
- **superseded stacked renderer** -> **`outputs/demo_goal.mp4`** (17.9 MB, 960x750, 25 s, 625
  frames). Pre-rendered so nothing can fail at the podium: plays in any player, no
  Python/GPU/internet. Herrera goal, test split, YOLO boxes + encoder input +
  generated commentary. Verified: broadcast scoreboard reads MUN 1-0 at 29:56,
  confirming timestamp alignment.
- **YOLO viability at 224p (measured, not assumed):** 12-13 persons detected around
  the goal, median confidence 0.54-0.57, box heights 16-40 px. Upscaling 2x gained
  nothing (13 vs 13), so **720p is unnecessary**. Weights cached at `./yolov8n.pt`
  for offline use.
- **Deliberately NOT built: a video-upload UI.** It would advertise a capability that
  does not exist (no event detection, no player identification), inviting exactly the
  demo that fails. Fixed clip + real annotations only.

### 4.11 Demo boundary — enforce this in the presentation

```
OUR MODEL    : structured event/context -> Transformer -> commentary
DEMO ASSEMBLY: commentary -> TTS -> video frame -> Whisper ASR
```

`render_demo.py` uses pretrained off-the-shelf tools and is **not** part of the
Transformer. Never blur this.

---

## 5. Workflow — execute in order, each stage gated by a real run

Nothing proceeds until the previous stage's check has actually printed the expected
thing.

| # | Stage | Artifact | Gate (must actually run) |
|---|---|---|---|
| 0 | Prep | `CLAUDE.md`, `requirements.txt`, `.gitignore`, dirs | **PASS** |
| 1 | Download | `download_and_parse.py` | **PASS** — all 471 games, 0 failures, `data/splits.json` written |
| 2 | Parse / preprocess | `data/processed.json` | **PASS (dev slice)** — 20 games, 1,780 examples, hash resolution **1914/1914 = 100%**, 0 leftover tags |
| 3 | Vocab | `data/vocab.json` | **PASS (dev slice)** — 1,176 types, `<UNK>` 0.44%, round-trip **3338/3338 identical**, 0 leaked entity tags |
| 4 | Model | `model.py` | **PASS** — 7,977,112 params; logits (4,17,1176); loss 7.28 vs ln(1176)=7.07; causal mask verified |
| 5 | Overfit test | — | **PASS** — 32 examples, loss **6.893 -> 0.0045**, greedy decode reproduces targets verbatim |
| 6 | Train | `train.py`, `checkpoints/` | **PASS** — 10/10 epochs, val **1.3245 -> 0.4097** (ppl 1.51), max epoch **258.2s** < 300s cap, checkpoint every epoch |
| 7 | Generate | `generate.py`, `outputs/` | **PASS** — greedy decode on official test split, 8 grounded triples in `outputs/generated_test.json` |
| 8 | BLEU-4 | `outputs/test_full_eval.json` | **PASS** — BLEU-4 **13.25** on all 7,826 official test examples (p1=50.9, p2=27.4, p3=18.3, p4=11.4) |
| 9 | Demo | `demo_live.py`, `demo_render.py` | **PASS** — live UI (stdlib http.server, 40 verified test events) + pre-rendered `outputs/demo_goal.mp4` (960x750, 25s, YOLO boxes + generated commentary). Inference only; model/training untouched. |

Phase 7 / later (explicitly NOT tonight): `Labels-v2.json` temporal join, role
grounding via lineup slot + tactic, FOOTPASS, jersey/tracking, video features,
SoccerNet-Echoes pretraining, BPE, beam / top-k, full BLEU / ROUGE / METEOR, larger
model, proper tests.

---

## 6. Implementation rules

1. **The downloaded files are authoritative.** If anything contradicts section 3,
   re-inspect the real JSON and update section 3. Never invent a field.
2. **Never let a real player or team name into the target vocabulary.** This is the
   anti-hallucination guarantee; it is structural, not statistical.
3. Always no-op `SoccerNet.Downloader.report` before downloading (section 3.2).
4. Verify by running. No stage is "done" because the code looks right.
5. Print shapes and losses at every new tensor path.
6. Do not train long before the overfit test passes (stage 5).
7. Keep the repo flat and small — no packaging, no premature abstraction.
8. Save vocab and model config inside the checkpoint so `generate.py` stands alone.
9. Windows: paths from `getListGames` use `\` separators; always go through
   `os.path.join` / `pathlib`. Read and write JSON with `encoding="utf-8"`.
10. If blocked, report the exact evidence found — do not guess around it.
11. **Split integrity is binding (4.10).** Use `getListGames(split, task="caption")` /
    `data/splits.json` as the only source of split membership. Never reshuffle, never
    merge, never train on valid or test. The test split stays unseen until final
    evaluation.
12. **Respect the training budget (4.12).** Checkpoint every epoch; stop and report if
    an epoch exceeds 5 minutes rather than trimming the model or the data to fit.

---

## 7. Repo layout

```
NLP Project/
|- CLAUDE.md               # this file
|- requirements.txt
|- .gitignore
|- download_and_parse.py   # stage 1-2  (to build)
|- vocab.py                # stage 3    (to build)
|- model.py                # stage 4    (to build)
|- train.py                # stage 6    (to build)
|- generate.py             # stage 7    (to build)
|- render_demo.py          # stage 9    (optional, to build)
|- data/
|   \- caption-2024/<league>/<season>/<game>/Labels-caption.json
|- checkpoints/
\- outputs/
```

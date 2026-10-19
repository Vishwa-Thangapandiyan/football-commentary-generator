# Scores

Re-run 2026-10-10 with `python eval_v2.py` and `python test_grounding.py`.
Test set: official SoccerNet test split, 7,826 examples. Same references for both models.

## Main scores

| Metric | V1 (`full_best.pt`) | V2 (`v2_best.pt`) |
|---|---|---|
| **BLEU-4** | **13.25** | **21.43** |
| p1 (unigram precision) | 50.9 | 46.2 |
| p2 (bigram precision) | 27.4 | 24.4 |
| p3 (trigram precision) | 18.3 | 16.6 |
| p4 (4-gram precision) | 11.4 | 11.3 |
| Brevity penalty | 0.570 | 1.000 |
| Output length / reference length (tokens) | 127,356 / 198,959 | 206,350 / 198,959 |
| Val loss (best epoch, 10) | 0.4097 | 0.3732 |
| Val perplexity | 1.51 | 1.45 |
| Ungrounded player tags, constrained (shipped) | 0.0% | 0.0% |
| Ungrounded player tags, unconstrained (debug) | 0.115% | 0.026% |
| Raw entity tags leaked | 0 | 0 |

p1-p4 are BLEU's n-gram precisions (%), not statistical p-values.
BLEU-4 = BP x geometric mean(p1..p4).

V1 has higher p1-p4 but lower BLEU because its output is too short.
That gives it a 0.57 brevity penalty.

## Training loss per epoch

| Epoch | V1 train | V1 val | V2 train | V2 val |
|---|---|---|---|---|
| 1 | 2.4546 | 1.3245 | 2.4460 | 1.3043 |
| 2 | 1.1532 | 0.8285 | 1.1289 | 0.7932 |
| 3 | 0.8143 | 0.6351 | 0.7813 | 0.5992 |
| 4 | 0.6484 | 0.5463 | 0.6159 | 0.5062 |
| 5 | 0.5569 | 0.4898 | 0.5203 | 0.4510 |
| 6 | 0.4948 | 0.4619 | 0.4598 | 0.4273 |
| 7 | 0.4544 | 0.4437 | 0.4170 | 0.4029 |
| 8 | 0.4227 | 0.4305 | 0.3866 | 0.3917 |
| 9 | 0.3993 | 0.4211 | 0.3615 | 0.3793 |
| 10 | 0.3816 | 0.4097 | 0.3439 | 0.3732 |

Source: `outputs/history_full.json`, `outputs/history_v2.json`. Training was not re-run.

## Grounding test (`test_grounding.py`, overfit checkpoint, 200 dev examples)

| Check | Result |
|---|---|
| Complete team names emittable | 0 |
| Banned tags picked under +1e9 adversarial logits (33,988 pairs) | 0 |
| Allowed tags still selectable | yes |
| Ungrounded rate, unconstrained | 28.50% |
| Ungrounded rate, constrained | 0.00% |
| Overall | **5/5 PASS** |

The 28.50% comes from the overfit checkpoint (trained on 32 examples).
It is not the trained model's rate.

---

# What the numbers mean

**Setup:** V1 and V2 are the same Transformer: 8.75M parameters, 4 encoder + 4 decoder layers, 8 attention heads, 10 epochs.
**Only the input is different.**

## 1. BLEU-4: V2 (21.43) > V1 (13.25)

- **BLEU** compares the generated sentence with the real commentator's sentence. It counts word sequences that appear in both. The scale is 0–100, and higher means closer.
- **V1 problem:** 62% of rows have no event label (`<no_event>`). For those rows, passes, shots, fouls and saves all look the same to the model.
- So V1 falls back on one short, common sentence (e.g. "X is adjudged offside").
- **V2 fix:** those rows get an action type (pass, cross, shot, save, dribble, foul, offside, clearance).
- Clearer input means V2 writes a sentence that fits the event, at full length.

## 2. p1–p4: V1 > V2, yet BLEU: V2 > V1

- **p1** = % of generated single words that also appear in the real sentence.
  **p2** = word pairs. **p3** = 3 words in a row. **p4** = 4 words in a row.
- These are precision scores. They are **not** statistical p-values.
- V1's p's are higher because V1 writes **short** sentences. Fewer words means fewer wrong words.
- BLEU **punishes short output** with the brevity penalty (BP):
  - V1 wrote 127k words vs 199k in the real commentary (64% of the length). BP = 0.57, so its score is cut almost in half.
  - V2 wrote 206k words, about the right length. BP = 1.0, so there is no cut.
- Without the length penalty, V1 would score about 23.
- **So V2 wins on BLEU because it says enough. Its individual word choices are not more accurate than V1's.**

## 3. Val loss: V2 (0.3732) < V1 (0.4097). Lower is better.

- **Val loss** measures how well the model predicts the next word of commentary it never trained on.
- **Perplexity 1.45** means the model is, on average, about as unsure as choosing between ~1.5 words at each step.
- It is this low because football commentary is very repetitive: 2,516 different words across 963k training words.
- V2 is lower because its input tells it more about what the sentence will say.
- **Loss is low but BLEU is moderate.** Val loss is measured with the real previous words given. BLEU is measured when the model writes the whole sentence alone, and many different sentences can be correct for one event.

## 4. Hallucinated players: 0.0% for both

- **Ungrounded** = the model mentions a player who is not in the input.
- Without the block: V1 0.115% (9 of 7,826), V2 0.026% (2 of 7,826).
- With the block (final output): **0**. Any player tag not in the input gets score −∞ before the next word is picked.
- Real names are **never** in the model's vocabulary. They are filled in after generation from the team lineup.
  So the model cannot invent a real player name.

## 5. Grounding test: 5/5 PASS

- The test forces every banned player tag to the highest possible score (+1,000,000,000). The model still picked a banned tag **0 times** (out of 33,988 tries).
- **This makes the 0% a guarantee by design, not luck.**
- The 28.50% comes from a deliberately weak model trained on only 32 examples. It shows the block still works when the model misbehaves.

## 6. Limit to know

- V2's action type was found with keyword rules on the real commentary sentence (e.g. the word "offside" in the sentence gives the label `offside`). This was also done for the test set.
- In a real system, the action type would come from an event detector watching the match.
- **So V2's 21.43 is a best case.** It shows how much a clear event type helps. It is not the score of a fully automatic system.

## Key points

1. **V2 > V1 on BLEU (21.43 vs 13.25).** Clearer input ends the repeated template and gives full-length commentary.
2. **V1 > V2 on p1–p4.** V1's sentences are short and safe, but BLEU penalises that.
3. **V2 has lower val loss (0.373 vs 0.410).** It predicts real commentary better.
4. **0 hallucinated players** in final output. This is guaranteed by design and tested.
5. **V2's gain depends on knowing the action type.** Here that type came from the reference text.

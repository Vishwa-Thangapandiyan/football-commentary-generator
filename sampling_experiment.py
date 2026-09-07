"""
ISOLATED EXPERIMENT: does torch.multinomial sampling from the trained model's own
output distribution produce useful, novel commentary - or does it collapse?

Touches nothing. Does not import from or alter model.py's decoder, the checkpoint,
training, evaluation, or the UI. It reuses the model's existing encode/decode/out
as read-only building blocks and implements sampling here.

    python sampling_experiment.py
"""

import sys

import torch
import torch.nn.functional as F

from generate import build_banned_mask, load_model, restore_names
from spot_to_commentary import build_input
from train import CommentaryDataset, collate
from vocab import BOS_ID, EOS_ID, PAD_ID

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVENTS = [
    {"t": 9 * 60 + 20, "label": "corner", "spivak_class": "Corner"},
    {"t": 9 * 60 + 46, "label": "corner", "spivak_class": "Corner"},
    {"t": 14 * 60 + 17, "label": "corner", "spivak_class": "Corner"},
]
TEMPS = [0.7, 0.9, 1.0]
TOP_P = 0.9
N_SAMPLES = 5
HOME, AWAY = "Manchester United", "Arsenal"


@torch.no_grad()
def sample_decode(model, src, banned, max_len, temperature, top_p, seed):
    """Greedy's argmax replaced by torch.multinomial over the temperature-scaled,
    top-p-filtered distribution. The grounding mask is applied BEFORE the softmax,
    so illegal tokens have probability exactly 0 and can never be drawn."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    src_pad = src.eq(PAD_ID)
    memory = model.encode(src, src_pad)
    ys = torch.full((1, 1), BOS_ID, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        h = model.decode(ys, memory, src_pad, ys.eq(PAD_ID))
        logits = model.out(h[:, -1])
        if banned is not None:
            logits = logits.masked_fill(banned, float("-inf"))   # grounding intact
        probs = F.softmax(logits / temperature, dim=-1)

        # nucleus (top-p) filter
        sp, si = probs.sort(descending=True, dim=-1)
        cum = sp.cumsum(-1)
        keep = (cum - sp) < top_p          # always keeps at least the top token
        sp = sp * keep
        sp = sp / sp.sum(-1, keepdim=True)

        pick = torch.multinomial(sp.cpu(), 1, generator=g).to(src.device)
        nxt = si.gather(-1, pick)
        ys = torch.cat([ys, nxt], dim=1)
        if int(nxt) == EOS_ID:
            break
    return ys


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, ck = load_model("checkpoints/full_best.pt", device)
    print(f"model on {device} | vocab {len(vocab):,} | val_loss {ck['val_loss']:.4f} "
          f"(perplexity {torch.exp(torch.tensor(ck['val_loss'])):.2f})\n")

    for ev in EVENTS:
        src_text = build_input(ev, HOME, AWAY)
        ex = {"input": src_text, "target": "", "grounding": {}}
        ds = CommentaryDataset([ex], vocab)
        src, _ = collate([ds[0]])
        src = src.to(device)
        banned = build_banned_mask([ex], vocab, device)

        mm, ss = int(ev["t"] // 60), int(ev["t"] % 60)
        print("=" * 78)
        print(f"EVENT {mm:02d}:{ss:02d}  Corner")
        print(f"INPUT: {src_text}")

        greedy = vocab.decode(
            model.greedy_decode(src, max_len=ck["max_tgt"], banned=banned)[0].tolist())
        print(f"\n  GREEDY (unchanged default):\n    {restore_names(greedy, {})}")

        for temp in TEMPS:
            print(f"\n  --- temperature={temp}, top_p={TOP_P} ---")
            seen = set()
            for i in range(N_SAMPLES):
                ids = sample_decode(model, src, banned, ck["max_tgt"],
                                    temp, TOP_P, seed=1000 + i)
                txt = restore_names(vocab.decode(ids[0].tolist()), {})
                seen.add(txt)
                flag = "  <-- same as greedy" if txt == greedy else ""
                print(f"    [{i + 1}] {txt}{flag}")
            print(f"    distinct: {len(seen)}/{N_SAMPLES}")
        print()


if __name__ == "__main__":
    main()

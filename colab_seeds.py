"""
Seed study on Colab: BDH vs Transformer, 3 seeds each, identical setup.

Official V2 splits, 10 epochs, checkpoint every epoch, then the full official test
split with the same BLEU-4 and grounding audit. Precision is set per GPU through
environment variables (dataset via DSET=v2|v3|v3_noevt, seeds via SEEDS=0,1,2):

  A100 (default)  GPU=a100  BDH bf16 (as upstream and the RTX 4060 run),
                            Transformer fp32 (as the original run)
  Tesla T4        GPU=t4 BDH_AMP=fp16 TF_AMP=fp16  (the T4 only emulates bf16)

BDH attention always runs in fp32 (see bdh_model.py), so no setting overflows.

Run from the local machine (the kernel stays busy, which keeps the VM alive):

    colab exec -s a100 --timeout 30000 -f colab_seeds.py

Resumable: each finished job leaves outputs/done_<job>, and is skipped on rerun.
Every job logs to outputs/colab_<job>.log; outputs/colab_status.txt has a summary.
"""

import os
import subprocess
import sys
import time

ROOT = "/content/proj"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
GPU = os.environ.get("GPU", "a100")
BDH_AMP = os.environ.get("BDH_AMP", "bf16")
TF_AMP = os.environ.get("TF_AMP", "fp32")
PY = sys.executable
# Dataset: v2 (keyword sub-labels; leaks the reference), v3 (SoccerNet-v2 actions),
# v3_noevt (v3 without the commentary's own <EVT> label).
DSET = os.environ.get("DSET", "v2")
VOCAB = {"v2": "v2", "v3": "v3", "v3_noevt": "v3_noevt"}[DSET]
PROC = f"data/processed_{DSET}.json"
VOC = f"data/vocab_{VOCAB}.json"
DATA = ["--data", PROC, "--vocab", VOC]

EVAL_TF = """
import json
from eval_v2 import score
res, recs = score(dict(name="Transformer", ckpt="checkpoints/{tag}_best.pt",
                    data="{proc}", vocab="{voc}"))
open("outputs/eval_{tag}_best_test.json", "w").write(
    json.dumps({{"summary": res, "records": recs}}, indent=1, ensure_ascii=False))
"""


def jobs():
    for s in SEEDS:
        tag = f"bdh_{DSET}_{GPU}_s{s}"
        yield tag, [PY, "train_bdh.py", "train", *DATA, "--tag", tag,
                    "--seed", str(s), "--amp", BDH_AMP]
        yield f"eval_{tag}", [PY, "train_bdh.py", "eval", "--ckpt",
                              f"checkpoints/{tag}_best.pt", "--data",
                              PROC, "--amp", BDH_AMP,
                              "--batch", "128", "--chunk", "1024",
                              "--mem-fraction", "0.9"]
        tag = f"tf_{DSET}_{GPU}_s{s}"
        yield tag, [PY, "train.py", "train", *DATA, "--tag", tag,
                    "--seed", str(s), "--amp", TF_AMP]
        yield f"eval_{tag}", [PY, "-c", EVAL_TF.format(tag=tag, proc=PROC, voc=VOC)]


def main():
    os.chdir(ROOT)
    os.makedirs("outputs", exist_ok=True)
    status = open("outputs/colab_status.txt", "a")
    for name, cmd in jobs():
        if os.path.exists(f"outputs/done_{name}"):
            continue
        t0 = time.time()
        line = f"{time.strftime('%H:%M:%S')} START {name}"
        print(line, flush=True)
        status.write(line + "\n")
        status.flush()
        with open(f"outputs/colab_{name}.log", "a") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        line = (f"{time.strftime('%H:%M:%S')} {'DONE' if rc == 0 else 'FAILED'} "
                f"{name} rc={rc} ({(time.time() - t0) / 60:.1f} min)")
        print(line, flush=True)
        status.write(line + "\n")
        status.flush()
        if rc == 0:
            open(f"outputs/done_{name}", "w").write(time.ctime())
        else:
            # Do not run later jobs on top of a failure; report and stop.
            break


if __name__ == "__main__":
    main()

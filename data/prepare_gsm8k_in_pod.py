"""Build GSM8K in TRL's prompt/solution format at /tmp/gsm8k, inside the training container.

Used as the recipe `pre_script` on clusters without a shared volume (see docs/run-hyperpod-eks.md). It only needs
`datasets`, which the training image already has, and Hub access. Same output as prepare_gsm8k.py.
"""

from pathlib import Path

from datasets import load_dataset

PROMPT = "Reason step by step, and put your final answer within \\boxed{}."


def to_trl(e):
    return {
        "prompt": [{"role": "system", "content": PROMPT}, {"role": "user", "content": e["question"]}],
        "solution": e["answer"].split("####")[-1].strip().replace(",", ""),
    }


out = Path("/tmp/gsm8k")
out.mkdir(parents=True, exist_ok=True)
d = load_dataset("openai/gsm8k", "main").map(to_trl, remove_columns=["question", "answer"])
for split in ("train", "test"):
    d[split].to_json(out / f"{split}.jsonl")

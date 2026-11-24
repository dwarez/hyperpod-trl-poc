# /// script
# dependencies = ["datasets", "boto3"]
# ///
"""Convert openai/gsm8k to TRL's prompt/solution format.

Usage:
    uv run prepare_gsm8k.py s3://bucket/prefix     # SageMaker training jobs
    uv run prepare_gsm8k.py /fsx/path/gsm8k        # HyperPod (shared filesystem)
"""

import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from datasets import load_dataset

SYSTEM_PROMPT = "Reason step by step, and put your final answer within \\boxed{}."


def to_trl(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "solution": example["answer"].split("####")[-1].strip().replace(",", ""),
    }


def write_splits(dataset, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ("train", "test"):
        path = directory / f"{split}.jsonl"
        dataset[split].to_json(path)
        paths.append(path)
    return paths


def main(destination: str):
    dataset = load_dataset("openai/gsm8k", "main").map(to_trl, remove_columns=["question", "answer"])
    if not destination.startswith("s3://"):
        for path in write_splits(dataset, Path(destination)):
            print(path)
        return

    import boto3

    url = urlparse(destination)
    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as tmp:
        for path in write_splits(dataset, Path(tmp)):
            key = f"{url.path.strip('/')}/{path.name}"
            s3.upload_file(str(path), url.netloc, key)
            print(f"s3://{url.netloc}/{key}")


if __name__ == "__main__":
    main(sys.argv[1])

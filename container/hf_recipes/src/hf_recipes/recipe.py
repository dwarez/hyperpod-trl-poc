"""Load a HyperPod recipe with `run.model_type: trl` and turn it into a TRL config file.

The recipe's `training_config` section has two keys:

    algorithm: grpo   # TRL training script: sft, dpo, grpo, kto, reward, rloo
    args: {...}       # TRL config file content (script, training and model arguments)

Distributed strategies are TRL/transformers training arguments (`deepspeed`, `fsdp`, `fsdp_config`), so the
same recipe runs unchanged under any launcher that starts one process per GPU.
"""

import os
import tempfile
from dataclasses import dataclass

import yaml

ALGORITHMS = ("sft", "dpo", "grpo", "kto", "reward", "rloo")


@dataclass
class TrlRecipe:
    algorithm: str
    args: dict

    @classmethod
    def load(cls, path: str) -> "TrlRecipe":
        with open(path) as f:
            recipe = yaml.safe_load(f)
        model_type = (recipe.get("run") or {}).get("model_type")
        if model_type != "trl":
            raise ValueError(f"{path}: expected run.model_type 'trl', got {model_type!r}")
        training_config = recipe["training_config"]
        algorithm = training_config["algorithm"]
        if algorithm not in ALGORITHMS:
            raise ValueError(f"{path}: unsupported algorithm {algorithm!r}, expected one of {ALGORITHMS}")
        return cls(algorithm=algorithm, args=dict(training_config["args"]))

    def write_trl_config(self, output_dir: str) -> str:
        args = dict(self.args)
        args.setdefault("output_dir", output_dir)
        fd, path = tempfile.mkstemp(prefix=f"trl-{self.algorithm}-", suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(args, f)
        return path

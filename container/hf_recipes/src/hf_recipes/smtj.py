"""SageMaker training job launcher for `run.model_type: trl` recipes.

`ModelTrainer.from_recipe` mounts the recipe as the `recipe` input channel and sets the
`sagemaker_recipe_local_path` hyperparameter. SageMaker starts one container per instance, so this
script starts one worker per GPU with torchrun, using the topology SageMaker provides.
"""

import json
import os
import subprocess
import sys

HYPERPARAMETERS_PATH = "/opt/ml/input/config/hyperparameters.json"
RESOURCE_CONFIG_PATH = "/opt/ml/input/config/resourceconfig.json"
DEFAULT_RECIPE_PATH = "/opt/ml/input/data/recipe/recipe.yaml"
MASTER_PORT = "29500"


def recipe_path() -> str:
    with open(HYPERPARAMETERS_PATH) as f:
        hyperparameters = json.load(f)
    return hyperparameters.get("sagemaker_recipe_local_path", DEFAULT_RECIPE_PATH).strip('"')


def gpus_per_node() -> int:
    if "SM_NUM_GPUS" in os.environ:
        return int(os.environ["SM_NUM_GPUS"])
    import torch

    return torch.cuda.device_count()


def torchrun_command(recipe: str) -> list[str]:
    with open(RESOURCE_CONFIG_PATH) as f:
        resources = json.load(f)
    hosts = sorted(resources["hosts"])
    return [
        "torchrun",
        f"--nnodes={len(hosts)}",
        f"--node_rank={hosts.index(resources['current_host'])}",
        f"--nproc_per_node={gpus_per_node()}",
        f"--master_addr={hosts[0]}",
        f"--master_port={MASTER_PORT}",
        "-m",
        "hf_recipes.worker",
        f"--config-path={os.path.dirname(recipe)}",
        f"--config-name={os.path.basename(recipe)}",
    ]


def main() -> int:
    # This entrypoint bypasses the sagemaker-training toolkit, which normally exports SM_MODEL_DIR.
    os.environ.setdefault("SM_MODEL_DIR", "/opt/ml/model")
    command = torchrun_command(recipe_path())
    print(f"[hf-recipes] {' '.join(command)}", flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())

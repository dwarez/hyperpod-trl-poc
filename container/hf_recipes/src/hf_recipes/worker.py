"""Per-process entry point for `run.model_type: trl` recipes.

Started once per GPU by torchrun or hyperpodrun: from `hf_recipes.smtj` on SageMaker training jobs, from
`/opt/bin/scripts/train-script-gpu.sh` on HyperPod EKS, and from the Slurm train script that the HyperPod
recipes launcher generates. The recipe location is passed the same way as for the other recipe types:

    python -m hf_recipes.worker --config-path=/config --config-name=config.yaml
"""

import argparse
import importlib
import os
import sys

from hf_recipes.recipe import TrlRecipe


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hf_recipes.worker")
    parser.add_argument("--config-path", "--config-dir", dest="config_path", required=True)
    parser.add_argument("--config-name", required=True)
    return parser.parse_args(argv)


def default_output_dir() -> str:
    if "SM_MODEL_DIR" in os.environ:
        return os.environ["SM_MODEL_DIR"]
    raise ValueError("Set training_config.args.output_dir in the recipe (e.g. a path on the shared FSx volume)")


def uses_vllm_tensor_parallel(args: dict) -> bool:
    return bool(args.get("use_vllm")) and int(args.get("vllm_tensor_parallel_size", 1)) > 1


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    recipe = TrlRecipe.load(os.path.join(args.config_path, args.config_name))
    output_dir = recipe.args.get("output_dir") or default_output_dir()
    trl_config = recipe.write_trl_config(output_dir)

    # The SageMaker SDK torchrun driver exports NCCL_PROTO=simple, which breaks vLLM colocated generation.
    os.environ.pop("NCCL_PROTO", None)
    # vLLM's custom all-reduce (tensor parallel > 1) fails with the DLC default expandable_segments allocator.
    if uses_vllm_tensor_parallel(recipe.args) and "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")

    script = importlib.import_module(f"trl.scripts.{recipe.algorithm}")
    parsed = script.make_parser().parse_args_and_config(["--config", trl_config])
    script.main(*parsed)


if __name__ == "__main__":
    main()

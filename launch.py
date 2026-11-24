"""Launch a `model_type: trl` HyperPod recipe as a SageMaker training job.

Usage:
    python launch.py recipes/<recipe>.yaml \
        --instance-type ml.g6e.12xlarge \
        --image <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag> \
        --role arn:aws:iam::<account>:role/<sagemaker-execution-role> \
        --data s3://<bucket>/<prefix>/gsm8k/ \
        [--env KEY=VALUE ...]
"""

import argparse
import os
import tempfile

from omegaconf import DictConfig, OmegaConf
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.configs import Compute, InputData, StoppingCondition
from sagemaker.train.model_trainer import ModelTrainer


def resolve_recipe(path: str) -> str:
    """Resolve `${base_results_dir}` the way the HyperPod recipes launcher does; the SDK can't."""
    recipe = OmegaConf.load(path)
    assert isinstance(recipe, DictConfig)
    recipe.base_results_dir = "/opt/ml/model"
    OmegaConf.resolve(recipe)
    del recipe["base_results_dir"]
    fd, resolved = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    OmegaConf.save(recipe, resolved)
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--image", required=True, help="training image built from container/")
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--data", required=True, help="S3 prefix with train.jsonl and test.jsonl (mounted as the train channel)")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    args = parser.parse_args()

    environment = dict(item.split("=", 1) for item in args.env)
    if os.environ.get("HF_TOKEN"):
        environment["HF_TOKEN"] = os.environ["HF_TOKEN"]

    trainer = ModelTrainer.from_recipe(
        training_recipe=resolve_recipe(args.recipe),
        training_image=args.image,
        compute=Compute(instance_type=args.instance_type, volume_size_in_gb=200),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=24 * 3600),
        environment=environment or None,
        sagemaker_session=Session(),
        role=args.role,
        base_job_name="trl-recipe",
    )
    trainer.train(input_data_config=[InputData(channel_name="train", data_source=args.data)], wait=args.wait)


if __name__ == "__main__":
    main()

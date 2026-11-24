# Run on SageMaker training jobs

Requirements: the training image pushed to ECR ([how-it-works.md](how-it-works.md#building-the-image)), a SageMaker execution role, and AWS credentials for the account.

```bash
# SageMaker Python SDK with the trl recipe type
git clone https://github.com/aws/sagemaker-python-sdk.git ../sagemaker-python-sdk-trl
git -C ../sagemaker-python-sdk-trl checkout 5fe66474
git -C ../sagemaker-python-sdk-trl apply "$PWD/patches/sagemaker-python-sdk-trl-recipes.patch"
uv venv -p 3.12
uv pip install -e ../sagemaker-python-sdk-trl/sagemaker-core -e ../sagemaker-python-sdk-trl/sagemaker-train

# GSM8K in TRL's prompt/solution format
uv run data/prepare_gsm8k.py s3://<bucket>/<prefix>/gsm8k

# Smoke test (Qwen3-0.6B, 10 GRPO steps, 4 GPUs)
.venv/bin/python launch.py recipes/trl-grpo-rlvr-qwen-3-0-dot-6b-lora-smoke.yaml \
  --instance-type ml.g6e.12xlarge \
  --image <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag> \
  --role arn:aws:iam::<account>:role/<sagemaker-execution-role> \
  --data s3://<bucket>/<prefix>/gsm8k/

# Qwen3-8B GRPO, 2 epochs (about 3.7 h on ml.p4de.24xlarge)
.venv/bin/python launch.py recipes/trl-grpo-rlvr-qwen-3-8b-lora.yaml --instance-type ml.p4de.24xlarge \
  --image ... --role ... --data ...
```

`launch.py` calls `ModelTrainer.from_recipe` with the image and mounts the data prefix as the `train` channel, which the recipes read from `/opt/ml/input/data/train`. It resolves `${base_results_dir}` in the recipe first: the recipes launcher normally provides it, and the SDK does not. `--env KEY=VALUE` adds environment variables to the job.

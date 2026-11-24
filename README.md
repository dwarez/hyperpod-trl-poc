# TRL recipes for SageMaker HyperPod on the Hugging Face training DLC

A proof of concept for running [TRL](https://github.com/huggingface/trl) fine-tuning as [SageMaker HyperPod recipes](https://github.com/aws/sagemaker-hyperpod-recipes), on the Hugging Face PyTorch training DLC (`huggingface-pytorch-training`), through the same launch paths the existing LLMFT and VERL recipes use.

It adds a new recipe type, `run.model_type: trl`. A TRL recipe is a regular HyperPod recipe whose `training_config` holds a TRL configuration, so any TRL technique (SFT, DPO, GRPO, KTO, reward modeling, ...) and any model supported by transformers can become a recipe without a model-specific training image.

## Status

| Launch path | Tested | Result |
|---|---|---|
| SageMaker training jobs (`ModelTrainer.from_recipe`) | Qwen3-0.6B GRPO smoke; Qwen3-8B GRPO, 2 epochs on ml.p4de.24xlarge | completed |
| HyperPod Slurm (recipes launcher, `cluster=slurm`) | Qwen3-0.6B GRPO smoke with vLLM, 1x H100, enroot/pyxis | completed |
| HyperPod EKS (recipes launcher, `cluster=k8s`) | Qwen3-0.6B GRPO smoke with vLLM, ml.g5.12xlarge, Kubeflow `PyTorchJob` | succeeded |

Not tested yet: `HyperPodPyTorchJob` with `hyperpodrun` (the test cluster had no HyperPod training operator), multi-node runs, and the launcher's `cluster=sm_jobs` mode. Details, numbers and the issues found along the way: [docs/results.md](docs/results.md).

## Where to look

| I want to... | Read |
|---|---|
| understand the design: recipe format, what changes in the SDK, the launcher and the container | [docs/how-it-works.md](docs/how-it-works.md) |
| run a recipe as a SageMaker training job | [docs/run-sagemaker.md](docs/run-sagemaker.md) |
| run a recipe on HyperPod EKS | [docs/run-hyperpod-eks.md](docs/run-hyperpod-eks.md) |
| run a recipe on HyperPod Slurm | [docs/run-hyperpod-slurm.md](docs/run-hyperpod-slurm.md) |
| see the results and the issues found | [docs/results.md](docs/results.md) |

## Repository layout

```
container/                 training image: the HF training DLC plus the recipe runtime
  hf_recipes/              Python package: recipe loader, per-GPU worker, SageMaker launcher
  scripts/                 train-script-gpu.sh, the EKS entry point the recipes chart calls
  sagemaker_entrypoint.sh  DLC entrypoint, extended to recognize recipe jobs
  Dockerfile
recipes/                   example TRL recipes (Qwen3-0.6B smoke, Qwen3-8B GRPO)
patches/                   changes to aws/sagemaker-python-sdk and aws/sagemaker-hyperpod-recipes
data/                      GSM8K preparation (to S3 or a shared filesystem, or inside the pod)
scripts/                   minimal reproduction of a vLLM bug found during the tests
launch.py                  submits a recipe as a SageMaker training job
docs/
```

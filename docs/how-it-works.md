# How it works

## The recipe format

A TRL recipe is a regular HyperPod recipe. `run.model_type: trl` selects the TRL runtime, and `training_config` holds two keys:

- `algorithm`: the TRL training script to run (`sft`, `dpo`, `grpo`, `kto`, `reward`, `rloo`);
- `args`: a TRL configuration, the same content as a `trl <algorithm> --config <file>` YAML (script, training and model arguments).

```yaml
display_name: Qwen 3 8B GRPO RLVR Fine-Tuning with LoRA (TRL)
version: 0.1.0
instance_types: ["ml.p5.48xlarge", "ml.p4de.24xlarge"]

run:
  name: trl-grpo-rlvr-qwen-3-8b-lora
  model_type: trl
  results_dir: ${base_results_dir}/${.name}

trainer:
  num_nodes: 1
  devices: 8

training_config:
  algorithm: grpo
  args:
    model_name_or_path: Qwen/Qwen3-8B
    dataset_name: /opt/ml/input/data/train
    reward_funcs: [accuracy_reward]
    use_peft: true
    lora_r: 32
    lora_alpha: 64
    learning_rate: 1.0e-5
    use_vllm: true
    vllm_mode: colocate
    # ... any other TRL GRPOConfig / ModelConfig field
```

Distributed strategies are TRL/transformers training arguments (`deepspeed`, `fsdp`, `fsdp_config`), so the same recipe runs unchanged on every launch path. The example recipes are in [`recipes/`](../recipes); the Qwen3-8B one uses the hyperparameters of `verl-grpo-rlvr-qwen-3-8b-lora.yaml` from the recipes repository.

## One runtime, three launch paths

Every launch path ends the same way: one process per GPU runs

```
python -m hf_recipes.worker --config-path=<dir> --config-name=<recipe>.yaml
```

which loads the recipe and runs the TRL script for `training_config.algorithm` with `training_config.args`. This mirrors the LLMFT image, whose processes run `-m amzn_awsllm_fine_tuning.train_hp --config-dir ... --config-name ...`.

| Launch path | Who starts the per-GPU workers |
|---|---|
| SageMaker training jobs | `ModelTrainer.from_recipe` uploads the recipe as the `recipe` input channel and sets the `sagemaker_recipe_local_path` hyperparameter. The image entrypoint sees it and runs `hf_recipes.smtj`, which starts torchrun with the topology from `/opt/ml/input/config/resourceconfig.json` |
| HyperPod EKS | the recipes launcher renders a Helm chart (`HyperPodPyTorchJob` or Kubeflow `PyTorchJob`) with the recipe mounted at `/config/config.yaml` and the container command `/opt/bin/scripts/train-script-gpu.sh`. Our script follows the same contract as the LLMFT image's: `SCRIPT_PATH`/`SCRIPT_ARGS`, `NODES`/`NTASKS_PER_NODE`, `PRE_SCRIPT_COMMANDS`/`POST_SCRIPT_COMMANDS`, and `hyperpodrun` instead of torchrun when `USE_HYPERPODRUN=true` |
| HyperPod Slurm | the recipes launcher writes `train_script.sh`, which runs `torchrun ... -m hf_recipes.worker --config-path=... --config-name=...` inside the container (docker or enroot) |

## What changes where

| Component | Change | Where |
|---|---|---|
| Training image | `hf_recipes` package; `/opt/bin/scripts/train-script-gpu.sh`; `hyperpod-elastic-agent` (provides `hyperpodrun`); the DLC entrypoint routes recipe jobs to `hf_recipes.smtj` | [`container/`](../container), built on `huggingface-pytorch-training:2.11.0-transformers5.13.1-cu130-amzn2023` |
| SageMaker Python SDK | `_is_llmft_recipe` also accepts `run.model_type: trl`, so the SDK packages TRL recipes like LLMFT/VERL ones (recipe channel, image entrypoint) | [`patches/sagemaker-python-sdk-trl-recipes.patch`](../patches/sagemaker-python-sdk-trl-recipes.patch), against `aws/sagemaker-python-sdk` master @ `5fe66474` |
| HyperPod recipes launcher | `trl` accepted by `main.py` and `stages.py` (no `launch_json` templatization yet), entry module `hf_recipes.worker`, no default git repository; the two example recipes under `recipes_collection/recipes/fine-tuning/trl/`; a fix for the indentation of `cluster.volumes` in the training chart | [`patches/sagemaker-hyperpod-recipes-trl.patch`](../patches/sagemaker-hyperpod-recipes-trl.patch), against `aws/sagemaker-hyperpod-recipes` main @ `aec331e` |

Without the SDK change, the stock SDK would also run a TRL recipe labelled `llm_finetuning_aws`, since it only uses `model_type` to choose the packaging path. A distinct type matters for the launcher (template processors, regional container maps), the validator and the catalog.

## Building the image

```bash
cd container
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --output type=registry,oci-mediatypes=false,name=<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag> .
```

`--provenance=false --sbom=false` and Docker media types keep the image importable by enroot (Slurm clusters with pyxis); see [results.md](results.md#issues-found).

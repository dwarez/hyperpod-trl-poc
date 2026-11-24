# Run on HyperPod Slurm

Tested on a HyperPod Slurm cluster with H100 nodes and enroot/pyxis (no Docker for users), on a shared GPU partition with one GPU per job.

The recipes launcher supports both container runtimes: with `container=<image URI>` it uses Docker on the nodes, with `container=/abs/path/image.sqsh` it uses enroot/pyxis. The commands below use enroot, which is what was tested.

## 1. Import the image as a squashfs file

enroot 3.4.1 could not import the image straight from ECR: its registry client fails on ECR's manifest response (`Could not process JSON input`), and BuildKit attestation manifests break it too. What worked: push the image without attestations ([how-it-works.md](how-it-works.md#building-the-image)), pull it with rootless podman inside a job, then convert it.

```bash
#!/bin/bash -l
# import_image.sh, run with: sbatch -p <cpu-partition> --cpus-per-task=16 --mem=31G import_image.sh
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$USER && mkdir -p $XDG_RUNTIME_DIR
IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
podman login --username AWS --password "$ECR_PASSWORD" ${IMAGE%%/*}   # ECR_PASSWORD=$(aws ecr get-login-password)
podman pull $IMAGE
enroot import -o /fsx/$USER/images/hyperpod-trl.sqsh podman://$IMAGE
```

With Docker on the nodes, skip this step: the generated `launch_docker_container.sh` logs in to ECR with the node's IAM role, which needs pull access to the repository.

## 2. Prepare the data and the launcher

On the login node:

```bash
BASE=/fsx/$USER/hyperpod-trl
uv run data/prepare_gsm8k.py $BASE/gsm8k

git clone https://github.com/aws/sagemaker-hyperpod-recipes.git
git -C sagemaker-hyperpod-recipes checkout aec331e
git -C sagemaker-hyperpod-recipes apply "$PWD/patches/sagemaker-hyperpod-recipes-trl.patch"
cd sagemaker-hyperpod-recipes
uv venv -p 3.12
# zstandard==0.15.2 (pinned in requirements.txt) has no wheels and needs a C compiler
echo "zstandard>=0.22" > /tmp/overrides.txt
uv pip install -r requirements.txt --override /tmp/overrides.txt
```

## 3. Launch

```bash
.venv/bin/python main.py \
  recipes=fine-tuning/trl/trl-grpo-rlvr-qwen-3-0-dot-6b-lora-smoke \
  cluster=slurm cluster_type=slurm instance_type=ml.p5.48xlarge \
  base_results_dir=$BASE \
  container=/fsx/$USER/images/hyperpod-trl.sqsh \
  recipes.trainer.devices=1 \
  recipes.training_config.args.dataset_name=$BASE/gsm8k \
  +recipes.training_config.args.output_dir=$BASE/output/trl-grpo-smoke \
  "~cluster.exclusive" "~cluster.mem" \
  +cluster.partition=<partition> +cluster.gpus_per_node=1 +cluster.time=1:00:00 \
  "+cluster.container_mounts=[/fsx]" \
  +env_vars.VLLM_USE_FLASHINFER_SAMPLER=0
```

- `~cluster.exclusive` and `~cluster.mem` drop the launcher defaults (`--exclusive`, `--mem=0`) that request whole nodes. Keep them for dedicated nodes, and set `recipes.trainer.devices` and `cluster.gpus_per_node` to the GPU count.
- `VLLM_USE_FLASHINFER_SAMPLER=0` works around the missing cuRAND headers in the image on H100 ([results.md](results.md#issues-found)).
- In Docker mode, only `base_results_dir` is mounted into the container (`cluster.container_mounts` applies to enroot only), so keep data and outputs under it.

Job logs: `$BASE/<job>/log-*.out`.

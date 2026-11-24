# Results

All runs use GRPO with LoRA on GSM8K (TRL prompt/solution format, `accuracy_reward` from `math-verify`), with vLLM generating in the training processes (`vllm_mode: colocate`) unless noted.

| Recipe | Launch path, hardware | Outcome |
|---|---|---|
| Qwen3-0.6B smoke, 10 steps | SageMaker training job, ml.g6e.12xlarge (4x L40S) | completed; reward 0.25 at step 1, 0.69 at step 3; 17 s of training |
| same | SageMaker training job, ml.g5.12xlarge (4x A10G) | completed; 10 steps |
| Qwen3-8B, hyperparameters of `verl-grpo-rlvr-qwen-3-8b-lora.yaml`, vLLM tensor parallel 2 | SageMaker training job, ml.p4de.24xlarge (8x A100 80 GB) | completed 2 epochs (about 460 optimizer steps) in 3.66 h of training; reward 0.92 → 0.97; log-prob gap between vLLM and trainer 0.010; KL about 0.017 |
| Qwen3-0.6B smoke, transformers generation (`use_vllm: false`) | HyperPod Slurm, 1x H100, enroot | completed; 10 steps, 171 s of training |
| Qwen3-0.6B smoke, `steps_per_generation: 2` | HyperPod Slurm, 1x H100, enroot | completed; 10 steps, 9.3 s of training; reward 0 → 0.625 |
| Qwen3-0.6B smoke | HyperPod EKS, ml.g5.12xlarge (4x A10G), Kubeflow `PyTorchJob` | succeeded; 10 steps, 15.6 s of training; reward 0.31 to 0.56 |

GSM8K is too easy for Qwen3-8B to show learning: at epoch 1.7, 94% of prompt groups had all 8 completions with the same reward, which leaves GRPO no signal. The 8B run shows the setup is stable at scale; measuring learning needs a harder dataset (for example `trl-lib/DeepMath-103K`).

## Issues found

Most of these concern the training image and would apply to any recipe runtime built on it.

| Issue | Effect | Status / fix |
|---|---|---|
| vLLM crashes on H100 with `max_num_seqs` of 4 or less: `illegal memory access` in `flash_attn_varlen_func` during CUDA graph capture | GRPO with small generation batches fails at startup on H100 (not seen on A10G, L40S, A100). TRL sets `max_num_seqs = per_device_train_batch_size x steps_per_generation x vllm_tensor_parallel_size` | Reproduced with vLLM alone ([`scripts/vllm_small_max_num_seqs_repro.py`](../scripts/vllm_small_max_num_seqs_repro.py)) on vLLM 0.25, 0.31, 0.32 and 0.33; fixed in 0.34. TRL supports vLLM 0.34 from its next release (huggingface/trl#8418). Until then, keep that product at 8 or more (the smoke recipe sets `steps_per_generation: 2`) |
| The image lacks `curand.h` | vLLM's FlashInfer sampler cannot JIT-compile its kernels on H100 (`Ninja build failed`) | `VLLM_USE_FLASHINFER_SAMPLER=0`; the image should ship the cuRAND headers |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (DLC default) | vLLM with tensor parallelism > 1 crashes in its custom all-reduce (`custom_all_reduce.cuh:455 'invalid argument'`) | `hf_recipes.worker` drops the setting when a recipe uses vLLM tensor parallelism > 1 |
| enroot 3.4.1 cannot import from ECR | the image cannot be pulled directly on enroot/pyxis clusters | push without attestations and Docker media types, then pull with podman and `enroot import podman://` ([run-hyperpod-slurm.md](run-hyperpod-slurm.md)) |
| The torch 2.13 base image declares CUDA 13.3 | under enroot it does not start on drivers that support at most CUDA 13.2; torch inside is built for CUDA 13.0 | runs with `NVIDIA_DISABLE_REQUIRE=1` and `/usr/local/cuda/compat` on `LD_LIBRARY_PATH` (forward compatibility). On SageMaker the image entrypoint handles this; enroot does not run the entrypoint |
| `hyperpod-elastic-agent` 1.1.3 requires `torch<2.12` and `pip<25.3` | fine on the torch 2.11 image; blocks `hyperpodrun` on torch 2.13+ images | needs a relaxed pin |
| The recipes chart renders `cluster.volumes` entries with the wrong indentation | Helm rejects the manifest as soon as a hostPath volume is set | fixed in [`patches/sagemaker-hyperpod-recipes-trl.patch`](../patches/sagemaker-hyperpod-recipes-trl.patch) |
| `per_device_train_batch_size: 8` with vLLM at 60% of GPU memory | out of memory on 80 GB GPUs for Qwen3-8B | the 8B recipe uses 2 per device x 16 accumulation steps, which keeps the same 32 prompts per update and 128 per rollout as the VERL recipe |

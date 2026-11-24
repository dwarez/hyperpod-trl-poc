"""Minimal reproduction: vLLM crashes on H100 with a small max_num_seqs, using TRL's colocate settings.

    torchrun --nproc_per_node 1 vllm_small_max_num_seqs_repro.py 4   # illegal memory access during CUDA graph capture
    torchrun --nproc_per_node 1 vllm_small_max_num_seqs_repro.py 8   # works

Observed on 1x H100 80GB (driver 595.91, CUDA 13.2) with vLLM 0.25.0 (torch 2.11, cu130), with
VLLM_USE_FLASHINFER_SAMPLER=0; the same failure appears through TRL with vLLM 0.31.0 (torch 2.13).
Not observed on A10G, L40S or A100.
"""

import sys

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.3,
        max_model_len=768,
        max_num_seqs=int(sys.argv[1]),
        distributed_executor_backend="external_launcher",
        seed=0,
        max_num_batched_tokens=4096,
        logprobs_mode="processed_logprobs",
    )
    print(llm.generate(["2+2="], SamplingParams(max_tokens=8))[0].outputs[0].text)


if __name__ == "__main__":
    main()

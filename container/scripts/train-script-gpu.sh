#!/bin/bash
#
# HyperPod EKS entry point. The recipes launcher (aws/sagemaker-hyperpod-recipes) runs
# `/opt/bin/scripts/train-script-gpu.sh` in every pod of the PyTorchJob / HyperPodPyTorchJob and passes:
#   SCRIPT_PATH, SCRIPT_ARGS         what to run per process (e.g. "-m hf_recipes.worker", "--config-path=/config ...")
#   NODES, NTASKS_PER_NODE           topology
#   USE_HYPERPODRUN=true             HyperPodPyTorchJob: launch through hyperpodrun (auto-resume) instead of torchrun
#   GIT_REPO_URL, GIT_BRANCH, GIT_COMMIT, GIT_UPDATE_ADAPTER   optional code checkout
#   PRE_SCRIPT_COMMANDS, POST_SCRIPT_COMMANDS                  ';'-separated shell commands
set -ex

source /usr/local/bin/hf_optimizations.sh

WORKDIR=""
if [[ -n "${GIT_REPO_URL}" ]]; then
    WORKDIR="${HOME}/tmp/${HOSTNAME}"
    rm -rf "${WORKDIR}"
    git clone "${GIT_REPO_URL}" "${WORKDIR}"
    cd "${WORKDIR}"
    [[ -n "${GIT_BRANCH}" ]] && git checkout "${GIT_BRANCH}"
    if [[ -n "${GIT_COMMIT}" ]]; then
        git fetch origin "${GIT_COMMIT}"
        git reset --hard "${GIT_COMMIT}"
    fi
    [[ "${GIT_UPDATE_ADAPTER}" == "true" ]] && pip install . --force-reinstall --no-deps
    WORKDIR="${WORKDIR}/"
fi

run_commands() {
    local commands
    IFS=';' read -r -a commands <<< "$1"
    for command in "${commands[@]}"; do
        eval "${command}"
    done
}

if [[ -n "${PRE_SCRIPT_COMMANDS}" ]]; then
    run_commands "${PRE_SCRIPT_COMMANDS}"
fi

if [[ "${USE_HYPERPODRUN}" == "true" ]]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$(ip -o -4 route show to default | awk '{print $5}')}"
    hyperpodrun --nproc_per_node "${NTASKS_PER_NODE}" --server-log-level warning ${WORKDIR}${SCRIPT_PATH} ${SCRIPT_ARGS}
else
    DISTRIBUTED_ARGS="--nproc_per_node ${NTASKS_PER_NODE}"
    if [[ "${NODES:-1}" -gt 1 ]]; then
        DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --nnodes ${NODES} --rdzv_backend=c10d --rdzv_endpoint=${JOB_NAME}-worker-0"
    fi
    torchrun ${DISTRIBUTED_ARGS} ${WORKDIR}${SCRIPT_PATH} ${SCRIPT_ARGS}
fi

if [[ -n "${POST_SCRIPT_COMMANDS}" ]]; then
    run_commands "${POST_SCRIPT_COMMANDS}"
fi

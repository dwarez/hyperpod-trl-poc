#!/usr/bin/env bash
#
# Entrypoint for the SageMaker training (sagemaker) target.
#
# Layers HF's training performance defaults on top of the AWS base contract,
# then hands off to the stock SageMaker training launcher
# (start_with_right_hostname.sh -> sagemaker-training toolkit), preserving the
# base's multi-node hostname workaround. Anything the user sets in their
# estimator / environment always wins.
#
# Jobs started from a HyperPod recipe (ModelTrainer.from_recipe) carry the
# sagemaker_recipe_local_path hyperparameter and run through hf_recipes.smtj.

# HF auto-optimization layer (sets training env defaults).
source /usr/local/bin/hf_optimizations.sh

if [[ "${1:-}" == "train" ]] && grep -qs '"sagemaker_recipe_local_path"' /opt/ml/input/config/hyperparameters.json; then
  exec python -m hf_recipes.smtj
fi

# Hand off to the stock SageMaker training launcher. Fall back to exec'ing
# the args directly (e.g. plain `docker run ... python train.py`) when the
# launcher isn't present.
if command -v start_with_right_hostname.sh >/dev/null 2>&1; then
  exec bash -m start_with_right_hostname.sh "$@"
elif [[ -f /usr/local/bin/start_with_right_hostname.sh ]]; then
  exec bash -m /usr/local/bin/start_with_right_hostname.sh "$@"
fi
exec "$@"

ARG BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.11.0-transformers5.13.1-cu130-amzn2023
FROM ${BASE_IMAGE}

# hyperpod-elastic-agent provides hyperpodrun (HyperPodPyTorchJob auto-resume); it requires torch<2.12.
COPY hf_recipes /tmp/hf_recipes
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps /tmp/hf_recipes \
    && /opt/venv/bin/python -m pip install --no-cache-dir hyperpod-elastic-agent==1.1.3 \
    && rm -rf /tmp/hf_recipes

COPY --chmod=755 scripts/train-script-gpu.sh /opt/bin/scripts/train-script-gpu.sh
COPY --chmod=755 sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh

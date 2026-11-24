# Run on HyperPod EKS

Tested on a HyperPod EKS cluster with one ml.g5.12xlarge node, the Kubeflow training operator (no HyperPod training operator), no Kueue and no PVC.

## Cluster requirements

- `kubectl` access to a namespace (an EKS access entry with a namespace-scoped access policy) and `helm`.
- The namespace must allow privileged pods (`pod-security.kubernetes.io/enforce=privileged`). The recipes chart runs the container privileged (EFA) and mounts hostPaths (`/dev/shm`, `/var/log/aws/clusters`); under the `baseline` policy the pods are rejected.
- Something mounted at `/data`: the chart pipes the job output to `/data/training.log`. Either an FSx PVC (`+cluster.persistent_volume_claims.0.claimName=<pvc> +cluster.persistent_volume_claims.0.mountPath=data`) or a hostPath volume as below.
- The nodes must be able to pull the training image from ECR.
- With the HyperPod training operator installed, drop `cluster.use_hyperpod_pytorch_job=false` to get a `HyperPodPyTorchJob` launched through `hyperpodrun` (not tested yet).

## Launch

Without a shared volume, the recipe's `pre_script` builds GSM8K inside the pod from the Hugging Face Hub ([`data/prepare_gsm8k_in_pod.py`](../data/prepare_gsm8k_in_pod.py)). It is passed base64-encoded because the chart splits `pre_script` on `;`.

```bash
git clone https://github.com/aws/sagemaker-hyperpod-recipes.git ../sagemaker-hyperpod-recipes-trl
git -C ../sagemaker-hyperpod-recipes-trl checkout aec331e
git -C ../sagemaker-hyperpod-recipes-trl apply "$PWD/patches/sagemaker-hyperpod-recipes-trl.patch"
B64=$(base64 < data/prepare_gsm8k_in_pod.py | tr -d '\n')

cd ../sagemaker-hyperpod-recipes-trl
uv venv -p 3.12
# zstandard==0.15.2 (pinned in requirements.txt) has no wheels and needs a C compiler
echo "zstandard>=0.22" > /tmp/overrides.txt
uv pip install -r requirements.txt --override /tmp/overrides.txt

.venv/bin/python main.py \
  recipes=fine-tuning/trl/trl-grpo-rlvr-qwen-3-0-dot-6b-lora-smoke \
  cluster=k8s cluster_type=k8s instance_type=ml.g5.12xlarge \
  base_results_dir=$PWD/results \
  container=<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag> \
  cluster.namespace=<namespace> \
  cluster.use_hyperpod_pytorch_job=false \
  cluster.persistent_volume_claims=null \
  "cluster.volumes=[{volumeName: data, hostPath: /tmp, mountPath: /data}]" \
  recipes.training_config.args.dataset_name=/tmp/gsm8k \
  +recipes.training_config.args.output_dir=/data/trl-grpo-smoke \
  "+recipes.pre_script=[echo $B64 | base64 -d | python]"
```

Add `dry_run=true` to only render the Helm chart (`results/<job>/k8s_template/`). Follow the job with `kubectl -n <namespace> get pytorchjobs` and `kubectl -n <namespace> logs <job>-worker-0`; remove it with `helm -n <namespace> uninstall <job>`.

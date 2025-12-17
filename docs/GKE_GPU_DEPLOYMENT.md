# GKE GPU Deployment Guide

This guide covers deploying candle-vllm to Google Kubernetes Engine (GKE) with GPU support.

## Prerequisites

- GKE cluster with GPU node pool
- `kubectl` configured to access the cluster
- OpenTofu/Terraform installed
- Docker for building images (optional, if rebuilding)

## CUDA and Driver Compatibility

The most common deployment issue is **CUDA version mismatch** between the container and the GPU driver on the node.

### Driver/CUDA Compatibility Matrix

| NVIDIA Driver | Max CUDA Version | GKE Default |
|---------------|------------------|-------------|
| 535.x         | CUDA 12.2        | Yes (2024)  |
| 545.x         | CUDA 12.3        | No          |
| 550.x         | CUDA 12.4        | No          |
| 555.x         | CUDA 12.5        | No          |
| 560.x         | CUDA 12.6        | No          |
| 565.x         | CUDA 12.8        | No          |

### Check Your GKE Driver Version

```bash
# Check driver version on GPU nodes
kubectl get nodes -l cloud.google.com/gke-accelerator -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | \
  xargs -I{} kubectl get node {} -o jsonpath='{.metadata.name}: {.status.nodeInfo.containerRuntimeVersion}{"\n"}'

# Or check via nvidia-smi in a pod
kubectl exec -n <namespace> <pod-name> -- cat /proc/driver/nvidia/version
```

### Common Error: CUDA_ERROR_UNSUPPORTED_PTX_VERSION

```
CUDA_ERROR_UNSUPPORTED_PTX_VERSION: "the provided PTX was compiled with an unsupported toolchain."
```

**Cause**: The container was built with a newer CUDA version than the driver supports.

**Solution**: Either:
1. Rebuild the container with a compatible CUDA version
2. Upgrade the GKE GPU drivers

## Building the Container

### For GKE with Driver 535.x (Default)

Use CUDA 12.2:

```bash
docker build -t your-registry/candle-vllm:cuda12.2-t4 \
  --build-arg CUDA_COMPUTE_CAP=75 \
  .
```

### For GKE with Driver 560+

Use CUDA 12.6 or 12.8:

```bash
# First update Dockerfile to use cuda:12.8.1-cudnn-*-ubuntu22.04 images
docker build -t your-registry/candle-vllm:cuda12.8 \
  --build-arg CUDA_COMPUTE_CAP=75 \
  .
```

### CUDA Compute Capability by GPU

Set `CUDA_COMPUTE_CAP` based on your target GPU:

| GPU             | Compute Cap | Architecture |
|-----------------|-------------|--------------|
| Tesla T4        | 75          | Turing       |
| A100            | 80          | Ampere       |
| A10G            | 86          | Ampere       |
| RTX 3090        | 86          | Ampere       |
| L4              | 89          | Ada Lovelace |
| RTX 4090        | 89          | Ada Lovelace |
| H100            | 90          | Hopper       |

## GKE Node Pool Configuration

### Verify GPU Node Pool

```bash
# List nodes with GPU labels
kubectl get nodes -L cloud.google.com/gke-nodepool -L cloud.google.com/gke-accelerator

# Check allocatable GPUs
kubectl get nodes -l cloud.google.com/gke-accelerator -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'

# Verify NVIDIA device plugin is running
kubectl get pods -n kube-system -l k8s-app=nvidia-gpu-device-plugin
```

### Expected Output

```
NAME                                         STATUS   ROLES    AGE   VERSION               GKE-NODEPOOL   GKE-ACCELERATOR
gke-cluster-ai-pool-gpu-xxxxx-xxxx           Ready    <none>   1d    v1.33.5-gke.1308000   ai-pool        nvidia-tesla-t4
```

## Terraform Deployment

### Required Variables

Create `deployment/k8s/gke/terraform.tfvars`:

```hcl
# GPU Node Pool - MUST match your actual node pool name
gpu_nodepool    = "ai-pool"           # Check with: kubectl get nodes -L cloud.google.com/gke-nodepool
gpu_accelerator = "nvidia-tesla-t4"   # Check with: kubectl get nodes -L cloud.google.com/gke-accelerator
gpu_count       = 1

# Required for MCP
tavily_api_key = "your-tavily-api-key"

# Optional: Tune for your GPU memory
# kvcache_mem_mb = 4096   # T4 has 16GB, can use more
# memory_limit   = "14Gi"
```

### Deploy

```bash
cd deployment/k8s/gke
tofu init
tofu plan
tofu apply
```

## Environment Variables

The deployment automatically sets these for GKE GPU compatibility:

| Variable | Value | Purpose |
|----------|-------|---------|
| `LD_LIBRARY_PATH` | `/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu` | GKE mounts NVIDIA driver at `/usr/local/nvidia` |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Set by NVIDIA device plugin |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Set by NVIDIA device plugin |

## Troubleshooting

### Pod Stuck in CrashLoopBackOff

1. **Check logs**:
   ```bash
   kubectl logs -n candle -l app.kubernetes.io/name=candle-vllm --tail=100
   ```

2. **Common errors**:
   - `Unable to dynamically load the "cuda" shared library`: Missing `LD_LIBRARY_PATH`
   - `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`: CUDA/driver version mismatch
   - `Model load failed: OOM`: Reduce `--mem` or use smaller model

### Verify GPU Access

```bash
# Exec into pod and check GPU
kubectl exec -n candle -l app.kubernetes.io/name=candle-vllm -- nvidia-smi

# Check environment
kubectl exec -n candle -l app.kubernetes.io/name=candle-vllm -- env | grep -E "(NVIDIA|CUDA|LD_LIBRARY)"
```

### Model Download Taking Too Long

The liveness probe may kill the pod before model download completes. The Terraform sets:

- `liveness_probe.initial_delay_seconds = 300` (5 minutes)
- `liveness_probe.failure_threshold = 5`

For very large models, you may need to increase these values.

### Persistent Model Cache

To avoid re-downloading models on pod restarts, use a PVC:

```hcl
hf_cache_pvc_name = "candle-hf-cache"  # Must exist in namespace
```

Create the PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: candle-hf-cache
  namespace: candle
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard-rwo
```

## Upgrading GKE GPU Drivers

If you need a newer CUDA version, upgrade the node pool drivers:

```bash
# Check current driver version
gcloud container node-pools describe <pool-name> \
  --cluster=<cluster-name> \
  --zone=<zone> \
  --format="value(config.accelerators)"

# Update to latest driver
gcloud container node-pools update <pool-name> \
  --cluster=<cluster-name> \
  --zone=<zone> \
  --accelerator type=nvidia-tesla-t4,count=1,gpu-driver-version=latest
```

Note: This will recreate nodes in the pool, causing brief downtime.

## References

- [NVIDIA CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [GKE GPU Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

# candle-vllm on GKE

This OpenTofu project manages both the GKE infrastructure (node pools) and the `candle-vllm` application deployment.

## Infrastructure Components

### GPU Node Pools (cluster.tf)
- **ai-pool** (primary): 1 node with NVIDIA Tesla T4 GPU (min=1, max=1)
- **ai-pool-2** (secondary): 1 node with NVIDIA Tesla T4 GPU (min=1, max=1)

Both node pools are configured with:
- Machine type: `n1-standard-4` (default)
- GPU: 1x NVIDIA Tesla T4
- Auto-repair and auto-upgrade enabled
- GPU taints to prevent non-GPU workloads

### Application Deployment (main.tf)
- `Deployment`, `Service`, and `Ingress` for `candle-vllm`
- (Optional) `Service` and `Ingress` for the built-in UI server on port `1999`
- A `ConfigMap` containing a minimal `models.yaml`
- (Optional) MCP configuration secret for tool calling

## Prerequisites

- GKE cluster already exists
- The Kubernetes namespace `candle` already exists
- A TLS secret for `https://candle-vllm.prometheusags.ai` already exists in that namespace
- GCP credentials configured (gcloud auth or service account)

## Setup and Configuration

1. Copy the example variables file:
```bash
cd deployment/k8s/gke
cp terraform.tfvars.example terraform.tfvars
```

2. Edit `terraform.tfvars` and set:
   - `gcp_project_id`: Your GCP project ID
   - `cluster_name`: Your GKE cluster name (default: "client-cluster")
   - `cluster_location`: Your cluster location (default: "us-central1-a")
   - `tavily_api_key`: Your Tavily API key (if using MCP)

## Deploying Node Pools and Application

### Initial Deployment

If you're deploying the node pools for the first time:

```bash
cd deployment/k8s/gke
tofu init
tofu apply
```

### Importing Existing Node Pool

If the `ai-pool` node pool already exists, you need to import it first:

```bash
# Import the existing ai-pool node pool
tofu import google_container_node_pool.gpu_primary projects/YOUR_PROJECT_ID/locations/LOCATION/clusters/CLUSTER_NAME/nodePools/ai-pool

# Then apply to update its configuration
tofu apply
```

### Verify Node Pools

To confirm the node pools and labels in your cluster:

```bash
# List node pools
gcloud container node-pools list --cluster=client-cluster --location=us-central1-a

# Check node labels
kubectl get nodes -L cloud.google.com/gke-nodepool -L cloud.google.com/gke-accelerator

# Describe a specific node pool
gcloud container node-pools describe ai-pool --cluster=client-cluster --location=us-central1-a
```

## Managing Node Pools

### Scaling Node Pools

To change the size of a node pool, modify the autoscaling settings in `cluster.tf`:

```hcl
autoscaling {
  min_node_count = 1  # Change these values
  max_node_count = 2  # as needed
}
```

Then apply:
```bash
tofu apply
```

### Switching Between Node Pools

To deploy your application to a specific node pool, update `gpu_nodepool` in `terraform.tfvars`:

```hcl
# Use primary pool
gpu_nodepool = "ai-pool"

# Or use secondary pool
gpu_nodepool = "ai-pool-2"
```

### Pinning to a Specific Node

To pin the deployment to a specific node within a pool:

1. Get the node name:
```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=ai-pool
```

2. Set `target_node_name` in `terraform.tfvars`:
```hcl
target_node_name = "gke-client-cluster-ai-pool-xxxxx"
```

3. Apply:
```bash
tofu apply
```

## GitHub Actions

The workflow `.github/workflows/deploy-gke.yaml`:
1. Builds and pushes `tribehealth/candle-vllm:latest`
2. Applies this OpenTofu project using a kubeconfig stored in GitHub Secrets

Required GitHub secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `KUBECONFIG_B64` (base64-encoded *raw* kubeconfig content; no `exec` auth plugins)
- `GCP_PROJECT_ID` (your GCP project ID)
- `TAVILY_API_KEY`

Optional secrets/vars (override defaults):
- `CANDLE_VLLM_TLS_SECRET_NAME` (defaults to `candle-tls`)
- `CANDLE_WEB_TLS_SECRET_NAME` (defaults to `CANDLE_VLLM_TLS_SECRET_NAME`)

## Troubleshooting

### Node Pool Already Exists Error

If you get an error that the node pool already exists, you need to import it:

```bash
tofu import google_container_node_pool.gpu_primary projects/YOUR_PROJECT_ID/locations/LOCATION/clusters/CLUSTER_NAME/nodePools/ai-pool
```

### GPU Not Available

Check that nodes have GPUs attached:

```bash
kubectl describe nodes -l cloud.google.com/gke-nodepool=ai-pool | grep nvidia.com/gpu
```

### Application Not Scheduling

Check node taints and tolerations:

```bash
# Check node taints
kubectl describe nodes -l cloud.google.com/gke-nodepool=ai-pool | grep Taints

# Check pod events
kubectl describe pod -n candle -l app.kubernetes.io/name=candle-vllm
```

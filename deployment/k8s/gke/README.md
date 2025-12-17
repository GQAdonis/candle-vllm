# candle-vllm on GKE (pre-existing cluster)

This OpenTofu project deploys `candle-vllm` into an existing GKE cluster.

Assumptions:
- The GKE cluster already exists.
- The Kubernetes namespace `candle` already exists.
- A TLS secret for `https://candle-vllm.prometheusags.ai` already exists in that namespace.
- There is a GPU node pool labeled `cloud.google.com/gke-nodepool=ai-pool-gpu` (T4).

This project creates:
- `Deployment`, `Service`, and `Ingress` for `candle-vllm`
- (Optional) `Service` and `Ingress` for the built-in UI server on port `1999`
- A `ConfigMap` containing a minimal `models.yaml`

## Apply locally

```bash
cd deployment/k8s/gke
tofu init
tofu apply
```

## GitHub Actions

The workflow `.github/workflows/deploy-gke.yaml`:
1. Builds and pushes `tribehealth/candle-vllm:latest`
2. Applies this OpenTofu project using a kubeconfig stored in GitHub Secrets

Required GitHub secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `KUBECONFIG_B64` (base64-encoded kubeconfig file content)
- `TAVILY_API_KEY`

Optional secrets/vars (override defaults):
- `CANDLE_VLLM_TLS_SECRET_NAME` (defaults to `candle-tls`)
- `CANDLE_WEB_TLS_SECRET_NAME` (defaults to `CANDLE_VLLM_TLS_SECRET_NAME`)

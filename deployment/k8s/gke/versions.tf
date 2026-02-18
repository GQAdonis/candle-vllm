terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
  }

  backend "kubernetes" {
    # Use local kubeconfig by default. If you prefer, override in CI with:
    # `tofu init -backend-config="config_path=/path/to/kubeconfig"`.
    config_path   = "~/.kube/config"
    namespace     = "candle"
    secret_suffix = "candle-vllm-tofu-state"
  }
}

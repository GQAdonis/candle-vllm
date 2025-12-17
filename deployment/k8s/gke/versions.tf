terraform {
  required_version = ">= 1.6.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
  }

  backend "kubernetes" {
    namespace     = "candle"
    secret_suffix = "candle-vllm-tofu-state"
  }
}

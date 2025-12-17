variable "kubeconfig_path" {
  type        = string
  description = "Path to kubeconfig used to authenticate to the cluster."
  default     = "~/.kube/config"
}

variable "namespace" {
  type        = string
  description = "Namespace to deploy into (must already exist)."
  default     = "candle"
}

variable "name" {
  type        = string
  description = "Base name for Kubernetes resources."
  default     = "candle-vllm"
}

variable "container_image" {
  type        = string
  description = "Container image to deploy."
  default     = "tribehealth/candle-vllm:latest"
}

variable "replicas" {
  type        = number
  description = "Number of replicas."
  default     = 1
}

variable "service_port" {
  type        = number
  description = "Service port."
  default     = 2000
}

variable "host" {
  type        = string
  description = "Ingress hostname."
  default     = "candle-vllm.prometheusags.ai"
}

variable "web_host" {
  type        = string
  description = "Ingress hostname for the UI server."
  default     = "candle-web.prometheusags.ai"
}

variable "tls_secret_name" {
  type        = string
  description = "Existing TLS secret name in the namespace."
  default     = "candle-tls"
}

variable "web_tls_secret_name" {
  type        = string
  description = "Existing TLS secret name for the UI ingress (defaults to tls_secret_name)."
  default     = ""
}

variable "ingress_class_name" {
  type        = string
  description = "Ingress class name (set to the controller in your cluster, e.g. nginx)."
  default     = "nginx"
}

variable "gpu_nodepool" {
  type        = string
  description = "GKE nodepool name for GPU scheduling."
  default     = "ai-pool-gpu"
}

variable "gpu_count" {
  type        = number
  description = "Number of GPUs to request/limit."
  default     = 1
}

variable "cpu_request" {
  type        = string
  description = "CPU request."
  default     = "500m"
}

variable "cpu_limit" {
  type        = string
  description = "CPU limit."
  default     = "4"
}

variable "memory_request" {
  type        = string
  description = "Memory request."
  default     = "2Gi"
}

variable "memory_limit" {
  type        = string
  description = "Memory limit."
  default     = "12Gi"
}

variable "kvcache_mem_mb" {
  type        = number
  description = "KV-cache memory in MB (passed as --mem)."
  default     = 2048
}

variable "max_num_seqs" {
  type        = number
  description = "Max sequences (passed as --max-num-seqs)."
  default     = 4
}

variable "enable_ui_server" {
  type        = bool
  description = "Whether to enable the built-in UI server on port 1999."
  default     = true
}

variable "ui_service_port" {
  type        = number
  description = "UI service port (must match the built-in UI port; with --p 2000, UI uses 1999)."
  default     = 1999
}

variable "hf_cache_pvc_name" {
  type        = string
  description = "Optional PVC name to persist HuggingFace cache; if empty, uses emptyDir."
  default     = ""
}

variable "enable_mcp" {
  type        = bool
  description = "Enable MCP servers (web/tool access) via a mounted mcp.json."
  default     = true
}

variable "tavily_api_key" {
  type        = string
  description = "Tavily API key used by the tavily MCP server."
  sensitive   = true
  default     = ""

  validation {
    condition     = (!var.enable_mcp) || (length(var.tavily_api_key) > 0)
    error_message = "tavily_api_key must be set when enable_mcp=true."
  }
}

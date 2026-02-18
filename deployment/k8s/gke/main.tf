locals {
  labels = {
    "app.kubernetes.io/name"      = var.name
    "app.kubernetes.io/part-of"   = "candle-vllm"
    "app.kubernetes.io/managed"   = "opentofu"
    "app.kubernetes.io/component" = "server"
  }

  node_selector = merge(
    var.gpu_nodepool != "" ? { "cloud.google.com/gke-nodepool" = var.gpu_nodepool } : {},
    var.gpu_accelerator != "" ? { "cloud.google.com/gke-accelerator" = var.gpu_accelerator } : {},
  )

  web_tls_secret_name = var.web_tls_secret_name != "" ? var.web_tls_secret_name : var.tls_secret_name
}

resource "kubernetes_config_map_v1" "models" {
  metadata {
    name      = "${var.name}-models"
    namespace = var.namespace
    labels    = local.labels
  }

  data = {
    "models.yaml" = <<-YAML
      idle_unload_secs: 3600
      default_model: llama-3.2-1b-instruct

      parking_lot:
        pool:
          worker_threads: 4
        limits:
          max_units: null
          max_queue_depth: 50
          timeout_secs: 300
        queue:
          backend: "memory"
          persistence: false
        mailbox:
          backend: "memory"
          retention_secs: 3600

      models:
        - name: llama-3.2-1b-instruct
          hf_id: bartowski/Llama-SmolTalk-3.2-1B-Instruct-GGUF
          weight_file: Llama-SmolTalk-3.2-1B-Instruct-Q4_K_M.gguf
          params:
            dtype: f16
            mem: ${var.kvcache_mem_mb}
            max_num_seqs: ${var.max_num_seqs}
            block_size: 64
            device_ids: [0]
            multithread: false
            prefill_chunk_size: 1024
          capabilities:
            vision_mode: disabled
    YAML
  }
}

resource "kubernetes_secret_v1" "mcp" {
  count = var.enable_mcp ? 1 : 0

  metadata {
    name      = "${var.name}-mcp"
    namespace = var.namespace
    labels    = local.labels
  }

  type = "Opaque"

  data = {
    "mcp.json" = base64encode(jsonencode({
      servers = [
        {
          name         = "tavily-mcp"
          url          = "https://mcp.tavily.com/mcp/?tavilyApiKey=${var.tavily_api_key}"
          timeout_secs = 30
          instructions = "Web search and retrieval via Tavily."
        },
      ]
    }))
  }
}

resource "kubernetes_deployment_v1" "app" {
  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
  }

  # Don't block `tofu apply` on rollout completion (model download/load can take
  # time and crashloops should be debugged via `kubectl logs`).
  wait_for_rollout = false

  spec {
    replicas = var.replicas

    selector {
      match_labels = {
        "app.kubernetes.io/name" = var.name
      }
    }

    template {
      metadata {
        labels = merge(local.labels, {
          "app.kubernetes.io/name" = var.name
        })
      }

      spec {
        node_selector = local.node_selector

        # Some clusters enforce scheduling constraints via admission policies that
        # require `spec.template.spec.affinity.nodeAffinity` rather than (or in
        # addition to) `nodeSelector`. Keep both, matching the GPU node pool.
        affinity {
          node_affinity {
            required_during_scheduling_ignored_during_execution {
              node_selector_term {
                dynamic "match_expressions" {
                  for_each = local.node_selector
                  content {
                    key      = match_expressions.key
                    operator = "In"
                    values   = [match_expressions.value]
                  }
                }

                # If target_node_name is set, pin to that specific node
                dynamic "match_expressions" {
                  for_each = var.target_node_name != "" ? [1] : []
                  content {
                    key      = "kubernetes.io/hostname"
                    operator = "In"
                    values   = [var.target_node_name]
                  }
                }
              }
            }
          }
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        container {
          name              = "candle-vllm"
          image             = var.container_image
          image_pull_policy = "Always"
          # The base CUDA image uses NVIDIA's entrypoint script. If we only set
          # `args`, Kubernetes will pass flags to that script and it will fail
          # with `exec: --: invalid option`. Set `command` to run the server
          # binary directly.
          command = ["/usr/local/bin/candle-vllm"]

          args = concat([
            "--h",
            "0.0.0.0",
            "--p",
            tostring(var.service_port),
            "--d",
            "0",
            "--mem",
            tostring(var.kvcache_mem_mb),
            "--max-num-seqs",
            tostring(var.max_num_seqs),
          ], var.enable_ui_server ? ["--ui-server"] : [])

          env {
            name  = "RUST_LOG"
            value = "info"
          }

          # GKE mounts NVIDIA driver libraries at /usr/local/nvidia/lib64
          env {
            name  = "LD_LIBRARY_PATH"
            value = "/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu"
          }

          env {
            name  = "CANDLE_VLLM_MODELS_CONFIG"
            value = "/config/models.yaml"
          }

          dynamic "env" {
            for_each = var.enable_mcp ? [1] : []
            content {
              name  = "CANDLE_VLLM_MCP_CONFIG"
              value = "/config/mcp.json"
            }
          }

          port {
            name           = "http"
            container_port = var.service_port
            protocol       = "TCP"
          }

          dynamic "port" {
            for_each = var.enable_ui_server ? [1] : []
            content {
              name           = "web"
              container_port = var.ui_service_port
              protocol       = "TCP"
            }
          }

          resources {
            requests = {
              cpu              = var.cpu_request
              memory           = var.memory_request
              "nvidia.com/gpu" = var.gpu_count
            }
            limits = {
              cpu              = var.cpu_limit
              memory           = var.memory_limit
              "nvidia.com/gpu" = var.gpu_count
            }
          }

          readiness_probe {
            http_get {
              path = "/v1/models"
              port = var.service_port
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            timeout_seconds       = 5
          }

          liveness_probe {
            http_get {
              path = "/v1/models"
              port = var.service_port
            }
            # Model download/load can take 2-5 minutes depending on network and model size
            initial_delay_seconds = 300
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 5
          }

          volume_mount {
            name       = "models-config"
            mount_path = "/config/models.yaml"
            sub_path   = "models.yaml"
            read_only  = true
          }

          dynamic "volume_mount" {
            for_each = var.enable_mcp ? [1] : []
            content {
              name       = "mcp-config"
              mount_path = "/config/mcp.json"
              sub_path   = "mcp.json"
              read_only  = true
            }
          }

          volume_mount {
            name       = "hf-cache"
            mount_path = "/data"
          }
        }

        volume {
          name = "models-config"
          config_map {
            name = kubernetes_config_map_v1.models.metadata[0].name
          }
        }

        dynamic "volume" {
          for_each = var.enable_mcp ? [1] : []
          content {
            name = "mcp-config"
            secret {
              secret_name = kubernetes_secret_v1.mcp[0].metadata[0].name
              items {
                key  = "mcp.json"
                path = "mcp.json"
              }
            }
          }
        }

        dynamic "volume" {
          for_each = var.hf_cache_pvc_name != "" ? [1] : []
          content {
            name = "hf-cache"
            persistent_volume_claim {
              claim_name = var.hf_cache_pvc_name
            }
          }
        }

        dynamic "volume" {
          for_each = var.hf_cache_pvc_name == "" ? [1] : []
          content {
            name = "hf-cache"
            empty_dir {}
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "svc" {
  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
  }

  spec {
    selector = {
      "app.kubernetes.io/name" = var.name
    }

    port {
      name        = "http"
      port        = var.service_port
      target_port = var.service_port
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_service_v1" "web" {
  count = var.enable_ui_server ? 1 : 0

  metadata {
    name      = "${var.name}-web"
    namespace = var.namespace
    labels    = local.labels
  }

  spec {
    selector = {
      "app.kubernetes.io/name" = var.name
    }

    port {
      name        = "http"
      port        = var.ui_service_port
      target_port = var.ui_service_port
      protocol    = "TCP"
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_ingress_v1" "ing" {
  metadata {
    name      = var.name
    namespace = var.namespace
    labels    = local.labels
    annotations = {
      "nginx.ingress.kubernetes.io/ssl-redirect"            = "true"
      "nginx.ingress.kubernetes.io/force-ssl-redirect"      = "true"
      "nginx.ingress.kubernetes.io/proxy-buffering"         = "off"
      "nginx.ingress.kubernetes.io/proxy-request-buffering" = "off"
      "nginx.ingress.kubernetes.io/proxy-read-timeout"      = "3600"
      "nginx.ingress.kubernetes.io/proxy-send-timeout"      = "3600"
    }
  }

  spec {
    ingress_class_name = var.ingress_class_name

    tls {
      hosts       = [var.host]
      secret_name = var.tls_secret_name
    }

    rule {
      host = var.host
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service_v1.svc.metadata[0].name
              port {
                number = var.service_port
              }
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_ingress_v1" "web" {
  count = var.enable_ui_server ? 1 : 0

  metadata {
    name      = "${var.name}-web"
    namespace = var.namespace
    labels    = local.labels
    annotations = {
      "nginx.ingress.kubernetes.io/ssl-redirect"            = "true"
      "nginx.ingress.kubernetes.io/force-ssl-redirect"      = "true"
      "nginx.ingress.kubernetes.io/proxy-buffering"         = "off"
      "nginx.ingress.kubernetes.io/proxy-request-buffering" = "off"
      "nginx.ingress.kubernetes.io/proxy-read-timeout"      = "3600"
      "nginx.ingress.kubernetes.io/proxy-send-timeout"      = "3600"
    }
  }

  spec {
    ingress_class_name = var.ingress_class_name

    tls {
      hosts       = [var.web_host]
      secret_name = local.web_tls_secret_name
    }

    rule {
      host = var.web_host
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service_v1.web[0].metadata[0].name
              port {
                number = var.ui_service_port
              }
            }
          }
        }
      }
    }
  }
}

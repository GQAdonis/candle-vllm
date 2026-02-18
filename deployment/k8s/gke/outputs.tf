output "service_name" {
  value = kubernetes_service_v1.svc.metadata[0].name
}

output "ingress_host" {
  value = var.host
}

output "gpu_primary_node_pool_name" {
  description = "Name of the primary GPU node pool"
  value       = google_container_node_pool.gpu_primary.name
}

output "gpu_secondary_node_pool_name" {
  description = "Name of the secondary GPU node pool"
  value       = google_container_node_pool.gpu_secondary.name
}

output "gpu_primary_node_pool_size" {
  description = "Current size of the primary GPU node pool"
  value       = google_container_node_pool.gpu_primary.node_count
}

output "gpu_secondary_node_pool_size" {
  description = "Current size of the secondary GPU node pool"
  value       = google_container_node_pool.gpu_secondary.node_count
}

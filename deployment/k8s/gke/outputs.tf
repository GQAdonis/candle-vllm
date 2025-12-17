output "service_name" {
  value = kubernetes_service_v1.svc.metadata[0].name
}

output "ingress_host" {
  value = var.host
}

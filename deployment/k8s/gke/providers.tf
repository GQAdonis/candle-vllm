provider "google" {
  project = var.gcp_project_id
  region  = var.cluster_location
}

provider "kubernetes" {
  config_path = pathexpand(var.kubeconfig_path)
}

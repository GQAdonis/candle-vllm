# GKE Cluster and Node Pool Management
#
# This file manages the GKE cluster infrastructure, including node pools.
# It assumes the cluster already exists and manages it via Terraform import.

# Data source to reference the existing GKE cluster
data "google_container_cluster" "existing" {
  name     = var.cluster_name
  location = var.cluster_location
  project  = var.gcp_project_id
}

# Primary GPU Node Pool (ai-pool) - Modified to size 1
resource "google_container_node_pool" "gpu_primary" {
  name       = "ai-pool"
  cluster    = data.google_container_cluster.existing.name
  location   = data.google_container_cluster.existing.location
  project    = var.gcp_project_id
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 1
  }

  node_config {
    machine_type = var.gpu_machine_type
    disk_size_gb = var.gpu_disk_size_gb
    disk_type    = "pd-standard"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      "node-pool"                       = "ai-pool"
      "workload"                        = "gpu"
      "cloud.google.com/gke-nodepool"   = "ai-pool"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  lifecycle {
    # Prevent accidental destruction of node pool
    prevent_destroy = false
    ignore_changes = [
      # Ignore changes to node count if autoscaling is managing it
      node_count,
    ]
  }
}

# Secondary GPU Node Pool (ai-pool-2) - New pool with size 1
resource "google_container_node_pool" "gpu_secondary" {
  name       = "ai-pool-2"
  cluster    = data.google_container_cluster.existing.name
  location   = data.google_container_cluster.existing.location
  project    = var.gcp_project_id
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 1
  }

  node_config {
    machine_type = var.gpu_machine_type
    disk_size_gb = var.gpu_disk_size_gb
    disk_type    = "pd-standard"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      "node-pool"                       = "ai-pool-2"
      "workload"                        = "gpu"
      "cloud.google.com/gke-nodepool"   = "ai-pool-2"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  lifecycle {
    # Prevent accidental destruction of node pool
    prevent_destroy = false
    ignore_changes = [
      # Ignore changes to node count if autoscaling is managing it
      node_count,
    ]
  }
}

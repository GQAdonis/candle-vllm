#!/bin/bash
#
# Build and push candle-vllm Docker image
# Configured for Tesla T4 GPU (compute capability 7.5)
#

set -e

IMAGE_TAG="tribehealth/candle-vllm:latest"

echo "Building image: ${IMAGE_TAG}"
echo "CUDA: 12.8.1"
echo "CUDA_COMPUTE_CAP: 75 (Tesla T4)"
echo ""

docker build \
  -t "${IMAGE_TAG}" \
  --build-arg CUDA_COMPUTE_CAP=75 \
  --no-cache \
  .

echo ""
echo "Build complete. Pushing to Docker Hub..."

docker push "${IMAGE_TAG}"

echo ""
echo "Done. Image pushed: ${IMAGE_TAG}"

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow Docker to connect to your X server
xhost +local:docker 2>/dev/null || true

cd "$SCRIPT_DIR"

case "${1:-run}" in
  build)
    echo "Building so101-moveit Docker image..."
    docker compose -f docker-compose.moveit.yml build
    ;;
  run)
    echo "Starting MoveIt + RViz (mock hardware)..."
    docker compose -f docker-compose.moveit.yml up
    ;;
  perception)
    echo "Starting blue-object detector + RViz (RealSense D435)..."
    docker compose -f docker-compose.moveit.yml run --rm perception
    ;;
  shell)
    echo "Opening shell inside the MoveIt container..."
    docker compose -f docker-compose.moveit.yml run --rm moveit bash
    ;;
  *)
    echo "Usage: $0 [build|run|perception|shell]"
    echo "  build       — build the Docker image"
    echo "  run         — launch MoveIt + RViz (default)"
    echo "  perception  — launch blue-object detector + RViz (RealSense D435)"
    echo "  shell       — open an interactive shell inside the container"
    exit 1
    ;;
esac

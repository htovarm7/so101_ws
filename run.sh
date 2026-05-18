#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

xhost +local:docker 2>/dev/null || true

case "${1:-moveit}" in
  build)
    echo "Building so101-moveit Docker image..."
    docker compose build
    ;;
  moveit)
    echo "Starting MoveIt + RViz (real hardware)..."
    docker compose up moveit
    ;;
  perception)
    echo "Starting arm + MoveIt + RealSense D435 + blue-object detector..."
    docker compose up perception
    ;;
  perception-only)
    echo "Starting RealSense D435 + blue-object detector (no arm)..."
    docker compose up perception-only
    ;;
  pick-and-place)
    echo "Starting MoveIt + RealSense + object/zone detection + pick-and-place..."
    docker compose up pick-and-place
    ;;
  pick-and-place-servo)
    echo "Starting Placo IK + RealSense + perception (Phase 2 verification)..."
    docker compose up pick-and-place-servo
    ;;
  shell)
    echo "Opening interactive shell inside the container..."
    docker compose run --rm shell
    ;;
  *)
    echo "Usage: $0 [build|moveit|perception|perception-only|pick-and-place|shell]"
    echo ""
    echo "  build             — build the Docker image"
    echo "  moveit            — MoveIt + RViz (real arm, no camera)"
    echo "  perception        — MoveIt + RealSense D435 + blue-object detector"
    echo "  perception-only   — RealSense D435 + blue-object detector (no arm)"
    echo "  pick-and-place    — full stack: MoveIt + RealSense + classifier + zone detector + sort_by_class"
    echo "  pick-and-place-servo — Placo IK + RealSense + perception (Phase 2; no MoveIt)"
    echo "  shell             — interactive shell inside the container"
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image=corsair-chair-pose:local
if ! docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=0
fi
docker build --force-rm -f "$root/docker/Dockerfile.pose" -t "$image" "$root"
mkdir -p "$root/results/pose-tables"
docker run --rm --user "$(id -u):$(id -g)" \
    -e CORSAIR_DEDUP_CORRESPONDENCES=1 \
    --mount "type=bind,src=$root,dst=/workspace,readonly" \
    --mount "type=bind,src=$root/results/pose-tables,dst=/workspace/results/pose-tables" \
    --entrypoint python "$image" -u reproduction/run_pose_tables.py --tables 1 "$@"

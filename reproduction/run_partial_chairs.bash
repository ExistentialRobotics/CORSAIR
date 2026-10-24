#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image=corsair-chair-pose:local
if ! docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=0
fi
docker build --force-rm -f "$root/docker/Dockerfile.pose" -t "$image" "$root"
containers=()
cleanup() {
    for container in "${containers[@]}"; do
        docker rm -v "$container" >/dev/null
    done
}
trap cleanup EXIT
mounts=()
for dataset in Scan2CAD_pc Scan2CAD_annotations ShapeNetCore.v2.PC15k; do
    docker pull "erlospm/data:$dataset"
    container="corsair-partial-${dataset//./-}-$$"
    docker create --name "$container" "erlospm/data:$dataset" >/dev/null
    containers+=("$container")
    mounts+=(--volumes-from "$container:ro")
done
mkdir -p "$root/results/partial-chairs"
docker run --rm --user "$(id -u):$(id -g)" \
    -e CORSAIR_DEDUP_CORRESPONDENCES=1 \
    --mount "type=bind,src=$root,dst=/workspace,readonly" \
    --mount "type=bind,src=$root/results/partial-chairs,dst=/workspace/results/partial-chairs" \
    "${mounts[@]}" --entrypoint python "$image" \
    -u reproduction/run_partial_chairs.py "$@"

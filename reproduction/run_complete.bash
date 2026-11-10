#!/usr/bin/env bash
# Fresh, reproducible evaluation and three audited deliverable tables.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image=corsair-chair-pose:local
if ! docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=0
fi
docker build --force-rm -f "$root/docker/Dockerfile.pose" -t "$image" "$root"
output="$root/results/complete/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$output"
docker image inspect "$image" > "$output/environment-image.json"
containers=()
cleanup() {
    for container in "${containers[@]}"; do
        docker rm -v "$container" >/dev/null
    done
}
trap cleanup EXIT
mounts=()
for dataset in Scan2CAD_pc Scan2CAD_annotations ShapeNetCore.v2.PC15k; do
    container="corsair-complete-${dataset//./-}-$$"
    docker create --name "$container" "erlospm/data:$dataset" >/dev/null
    containers+=("$container")
    mounts+=(--volumes-from "$container:ro")
    docker image inspect "erlospm/data:$dataset" > "$output/dataset-$dataset.json"
done
docker run --rm --user "$(id -u):$(id -g)" \
    -e CORSAIR_DEDUP_CORRESPONDENCES=1 \
    --mount "type=bind,src=$root,dst=/workspace,readonly" \
    --mount "type=bind,src=$output,dst=/deliverables" \
    "${mounts[@]}" --entrypoint python "$image" \
    -u reproduction/run_complete.py --output /deliverables "$@" 2>&1 | tee "$output/run.log"
echo "Deliverables: $output"

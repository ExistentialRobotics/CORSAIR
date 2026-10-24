#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image=corsair-chair-pose:local
if ! docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=0
fi
docker build --force-rm -f "$root/docker/Dockerfile.pose" -t "$image" "$root"
table_data="$root/data/ShapeNetCore.v2.PC15k/04379243"
mkdir -p "$table_data"
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
    --mount "type=bind,src=$root,dst=/workspace,readonly" \
    --mount "type=bind,src=$table_data,dst=/workspace/data/ShapeNetCore.v2.PC15k/04379243" \
    "$image" -u reproduction/fetch_table_cads.py
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
    if [[ "$dataset" == ShapeNetCore.v2.PC15k ]]; then
        docker run --rm --volumes-from "$container" --entrypoint mkdir "$image" \
            -p /data/ShapeNetCore.v2.PC15k/04379243
    fi
    mounts+=(--volumes-from "$container:ro")
done
mkdir -p "$root/results/pose-tables"
docker run --rm --user "$(id -u):$(id -g)" \
    -e CORSAIR_DEDUP_CORRESPONDENCES=1 \
    --mount "type=bind,src=$root,dst=/workspace,readonly" \
    --mount "type=bind,src=$root/results/pose-tables,dst=/workspace/results/pose-tables" \
    --mount "type=bind,src=$table_data,dst=/data/ShapeNetCore.v2.PC15k/04379243,readonly" \
    "${mounts[@]}" --entrypoint python "$image" \
    -u reproduction/run_pose_tables.py "$@"

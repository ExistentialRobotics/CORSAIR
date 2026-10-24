#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="corsair-chair-pose:local"
output="$root/results"
arguments=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            if [[ $# -lt 2 ]]; then
                echo "--output requires a directory" >&2
                exit 2
            fi
            output="$2"
            shift 2
            ;;
        --output=*)
            output="${1#*=}"
            shift
            ;;
        *)
            arguments+=("$1")
            shift
            ;;
    esac
done
if ! docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=0
fi
docker build --force-rm --file "$root/docker/Dockerfile.pose" --tag "$image" "$root"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
exec docker run --rm --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output,dst=/workspace/results" \
    "$image" "${arguments[@]}"

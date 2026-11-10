#!/usr/bin/env bash
# Run this on the HOST. No checkout or Python installation is required.
set -euo pipefail
started=$SECONDS
show_elapsed() {
    local elapsed=$((SECONDS - started))
    printf 'Elapsed: %02d:%02d:%02d' "$((elapsed / 3600))" "$((elapsed / 60 % 60))" "$((elapsed % 60))" >&2
}
image="${CORSAIR_IMAGE:-maniothelej/corsair-icp:repro-v1}"
output="${CORSAIR_OUTPUT:-$PWD/corsair-reproduction-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
cleanup() {
    local status=$?
    show_elapsed
    printf '\n' >&2
    return "$status"
}
trap cleanup EXIT
docker image inspect "$image" > "$output/app-image.json"
docker run --rm --init --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$output,dst=/results" \
    --entrypoint python "$image" \
    -u reproduction/reproduce.py --output /results "$@" 2>&1 | tee "$output/run.log"
echo "Results and logs: $output"

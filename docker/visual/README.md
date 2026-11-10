# CORSAIR + ICP in a browser

The standalone `corsair-icp:local` image includes the CPU inference runtime,
checkpoints, bundled ShapeNet chair/table test subsets, evaluation scripts, and
a browser viewer. No GPU, X server, or source-code bind mount is required.

## Evaluate complete chairs and tables (default)

The public image is `maniothelej/corsair-icp:repro-v1` (Linux amd64). A plain
`docker run` evaluates 100 complete ShapeNet chairs and 100 complete ShapeNet
tables with the correct CAD supplied, fresh inference, tqdm progress, and
reference verification. It exits when the evaluation finishes. Partial scans
are skipped.

```sh
docker pull maniothelej/corsair-icp:repro-v1
docker run --rm --init -v corsair-results:/results maniothelej/corsair-icp:repro-v1 --workers 12
```

Each run is saved in a new timestamped directory under `/results` in the
results volume.
The final metric is the combined CORSAIR + ICP chair/table joint success.
Detailed metrics for every method and category are saved in the CSV reports.
Use `--verbose` for detailed console output, or `--output /results/NAME` to
choose a fresh output directory. No dataset-image mounts or exposed ports are
needed.

## Start the optional browser

Add `serve` to start the browser server instead:

```sh
docker pull maniothelej/corsair-icp:repro-v1
docker run -d --name corsair-icp --init \
  -p 127.0.0.1:8080:8080 \
  -v corsair-results:/results \
  maniothelej/corsair-icp:repro-v1 serve
```

Open <http://localhost:8080> and click **Run evaluation**. The default evaluates
one complete chair and one complete table. The bundled complete-CAD evaluation
needs no additional downloads. Evaluating partial scans needs the dataset
mounts described below; use the public image name in place of
`corsair-icp:local` in that command.

## Reproduce the reported number

The Docker launcher evaluates 100 complete chairs and 100 complete tables from
bundled ShapeNet data. It skips partial scans and needs no dataset-image mounts.
The final CORSAIR + ICP result is the arithmetic mean of the chair and table
joint-success percentages (RRE ≤5° AND normalized RTE ≤0.02).

```sh
docker pull maniothelej/corsair-icp:repro-v1
docker run --rm --entrypoint cat maniothelej/corsair-icp:repro-v1 /workspace/docker/visual/reproduce.bash > reproduce-corsair.bash
bash reproduce-corsair.bash --workers 12
```

The saved reference has 97% for chairs and 93% for tables, so the expected final
metric is:

```text
CORSAIR + ICP: 95.00%
```

The evaluation shows tqdm bars with completed tasks, elapsed time, speed, and
estimated remaining time for each feature-extraction and registration phase.
The launcher also prints total elapsed time and the output directory. Add
`--verbose` to see per-case progress, separate category/method metrics, and the
verification JSON.

Each run performs fresh inference and registration. It compares the complete
chair/table summary metrics and all 600 per-case/method rows against the saved
reference, including query/CAD identities. Differences in summary percentages
or per-case pose errors beyond 1e-8 absolute cause a nonzero exit.
`reproduction-check.json` records the verification; a mismatch is always printed.
Full logs, detailed category/method CSVs, and pose archives remain in the new
`corsair-reproduction-TIMESTAMP` output directory. `summary.json` saves the
combined final result. The one-model browser default is a smoke test and does
not reproduce the aggregate percentage.

To check an existing run, invoke `reproduction/reproduce.py --check-only /results/RUN`
inside the image with the results directory mounted. Existing runs that include
partial scans are also checked against the known-CAD and retrieved-CAD references.
The browser can evaluate complete CAD pairs and partial scans together with the
mounts below.

## Build locally

From the repository root, build the runtime and then the app:

```sh
docker build -f docker/Dockerfile.pose -t corsair-chair-pose:local .
docker build -f docker/Dockerfile.visual -t corsair-icp:local .
docker run -d --name corsair-icp --init \
  -p 127.0.0.1:8080:8080 \
  -v corsair-results:/results \
  corsair-icp:local serve
```

Open <http://localhost:8080> and click **Run evaluation**. The default evaluates
one chair and one table with fresh inference. Increase the count up to 100
per category for a larger run. CPU runs can take tens of minutes or over an
hour depending on count, workers, and evaluation mode.

Select a case to compare input, RANSAC, CORSAIR, and CORSAIR + ICP. Drag to
rotate all four views and scroll to zoom. Teal points are the target and orange
points are the source under the selected transform. All panels share a camera.
The viewer samples at most 3,500 points per cloud; the evaluation uses the
original clouds. Links open the reports, CSVs, and protocol metadata. Raw
cloud/pose archives and inference logs stay in the results volume.

The bundled evaluation uses complete, noiseless synthetic pairs with the
correct CAD supplied. Full mode adds partial Scan2CAD known-CAD and retrieved-CAD
evaluation. ICP accepts a proposal only when geometric nearest-neighbor RMSE
improves. It does not use ground-truth pose error for acceptance, and improved
RMSE does not guarantee improved rotation or translation error.

The service runs one evaluation at a time. It persists run status and logs;
stopping the container interrupts active computation, and interrupted runs
are labeled when the service restarts. Completed results remain available.
The UI is intended for local use and the command binds only to localhost.

```sh
docker logs corsair-icp
docker stop corsair-icp
docker start corsair-icp
docker cp corsair-icp:/results ./corsair-results-export
```

With Docker Compose installed, `docker compose -f compose.visual.yml up -d`
starts the same app after the runtime has been built.

## Evaluate complete CAD pairs and partial scans in the browser

The full evaluation additionally needs the three existing dataset images.
Create data containers once; their dataset volumes are mounted read-only.
Stop and remove the app container before recreating it with these mounts.
The results volume survives container removal.

```sh
docker create --name corsair-shapenet erlospm/data:ShapeNetCore.v2.PC15k
docker create --name corsair-scans erlospm/data:Scan2CAD_pc
docker create --name corsair-annotations erlospm/data:Scan2CAD_annotations
docker stop corsair-icp
docker rm corsair-icp
docker run -d --name corsair-icp --init \
  -p 127.0.0.1:8080:8080 \
  -v corsair-results:/results \
  --volumes-from corsair-shapenet:ro \
  --volumes-from corsair-scans:ro \
  --volumes-from corsair-annotations:ro \
  corsair-icp:local serve
```

Choose the browser's full evaluation mode to run complete ShapeNet chair/table
pairs and partial Scan2CAD chairs with both known and top-1 retrieved CAD.
The results of each evaluation can be inspected in the same viewer. This performs
the existing audited `reproduction/run_complete.py` workflow inside the image.

## Image sharing

Only the final app image needs to be distributed; its runtime base is included
in its image layers. The public repository is
<https://hub.docker.com/r/maniothelej/corsair-icp>. Local build steps do not push
to the registry.
For offline transfer, use `docker save -o corsair-icp.tar corsair-icp:local`
and `docker load -i corsair-icp.tar` on the destination.

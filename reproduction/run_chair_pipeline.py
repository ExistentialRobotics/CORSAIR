"""Evaluate fresh CORSAIR chair predictions before and after ICP refinement."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.pose_metrics import pose_errors
from utils.pose_refinement import refine_pose_icp


def plot_comparison(frame, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9), facecolor="white")
    fig.suptitle(f"CORSAIR → ICP → scoring · {len(frame)} chair pairs", fontsize=19)
    ax = fig.add_subplot(2, 1, 1)
    fields = ["rotation_pass_percent", "translation_pass_percent", "joint_pass_percent"]
    x = np.arange(3)
    colors = ["#527a99", "#bd7654"]
    for i, stage in enumerate(["before", "after"]):
        values = [summary[stage][field] for field in fields]
        bars = ax.bar(x + (i - .5) * .34, values, .34, color=colors[i],
                      label="CORSAIR" if stage == "before" else f"CORSAIR + ICP (≤{summary['icp_iteration_cap']} iterations)")
        ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.axhline(50, ls="--", color="#555555", label="50% target")
    ax.set_xticks(x, ["RRE ≤ 5°", "RTE ≤ 0.02", "Both"])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Pairs (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for panel, field, threshold, label in [(3, "rre_deg", 5, "Rotation error (degrees)"),
                                          (4, "rte", .02, "Translation error (normalized units)")]:
        ax = fig.add_subplot(2, 2, panel)
        for i, stage in enumerate(["before", "after"]):
            ax.scatter(np.arange(1, len(frame) + 1), np.maximum(frame[f"{stage}_{field}"], 1e-16),
                       s=16, alpha=.75, color=colors[i], label=stage)
        ax.axhline(threshold, ls="--", color="#555555")
        ax.set_yscale("log")
        ax.set_xlabel("Chair index")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "Same coarse predictions before/after. Full noiseless CAD pairs and known object pairing.\n"
             "Float64 errors; log plots show tiny nonzero residuals. This is not real-scan end-to-end validation.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.82, bottom=.13, hspace=.45, wspace=.25)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="Fresh inference is always enabled")
    parser.add_argument("--models", type=int, default=100, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--icp", choices=["standard", "multiscale"], default="standard")
    parser.add_argument("--icp-iterations", type=int, default=10,
                        help="Positive standard ICP cap, or per-stage cap for multiscale (default: 10)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="Evaluation output directory")
    args = parser.parse_args()
    if args.icp_iterations < 1:
        parser.error("--icp-iterations must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"shapenet-seed0-chair-{args.models}-1"
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    baseline = output / "baseline" / tag
    baseline.mkdir(parents=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CORSAIR_DEDUP_CORRESPONDENCES="1", POSE_WORKERS=str(args.workers))
    command = [sys.executable, "-u", str(ROOT / "reproduction/run_pose.py"),
               "--shapenet-root", str(ROOT / "docker/data/ShapeNetCore.v2.PC15k"),
               "--category", "chair", "--model-ckpt", str(ROOT / "ckpts/scannet_ret_chair"),
               "--n-models", str(args.models), "--n-poses-per-model", "1",
               "--device", "cpu", "--random-seed", "0", "--icp-refinement", "none"]
    print("Running fresh CORSAIR inference.", flush=True)
    with (baseline / "inference.log").open("w") as log:
        subprocess.run(command, cwd=baseline, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    prediction_csv = baseline / f"results-{stem}.csv"
    prediction_archive = baseline / f"poses-{stem}.npz"
    print(f"Coarse predictions: {prediction_archive}", flush=True)
    coarse_hash = hashlib.sha256(prediction_archive.read_bytes()).hexdigest()
    data = pd.read_csv(prediction_csv)
    with np.load(prediction_archive) as stored:
        truth = stored["poses_gt"].copy()
        initial = stored["poses_pred_sym"].copy()
    if len(data) != args.models or initial.shape != (args.models, 4, 4):
        raise ValueError("Unexpected coarse prediction count or shape")

    def process(i):
        row = data.iloc[i]
        source = np.load(ROOT / "docker/data/ShapeNetCore.v2.PC15k/03001627/test" / row.model)
        source -= source.mean(0, keepdims=True)
        source /= np.linalg.norm(source, axis=1).max()
        target = source @ truth[i, :3, :3].T + truth[i, :3, 3]
        refined = refine_pose_icp(source, target, initial[i], multiscale=args.icp == "multiscale",
                                  max_iterations=args.icp_iterations)
        record = {"model": row.model, "symmetry_order": int(row.symmetry_label)}
        for stage, transform in [("before", initial[i]), ("after", refined)]:
            rre, rte = pose_errors(transform, truth[i], int(row.symmetry_label))
            record.update({f"{stage}_rre_deg": rre, f"{stage}_rte": rte,
                           f"{stage}_rotation_pass": rre <= 5,
                           f"{stage}_translation_pass": rte <= .02,
                           f"{stage}_joint_pass": rre <= 5 and rte <= .02})
        if not np.isclose(record["before_rte"], row.rte_sym, atol=1e-6):
            raise ValueError("Baseline translation error does not match the saved experiment")
        return record, refined, source, target

    records, refined_poses, observations = [], [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (record, refined, source, target) in enumerate(tqdm(
                pool.map(process, range(args.models)), total=args.models, desc="Same-prediction ICP")):
            records.append(record)
            refined_poses.append(refined)
            observations[f"source_{i}"] = source
            observations[f"target_{i}"] = target
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_chair.csv", index=False)
    np.savez_compressed(output / "poses.npz", model_ids=data.model.to_numpy(dtype=str),
                        ground_truth=truth, corsair_initial=initial, corsair_icp=np.stack(refined_poses))
    np.savez_compressed(output / "observations.npz", **observations)
    summary = {"queries": len(frame), "inference": "fresh",
               "random_seed": 0,
               "checkpoint_sha256": hashlib.sha256((ROOT / "ckpts/scannet_ret_chair").read_bytes()).hexdigest(),
               "runtime": {name: version(name) for name in ["numpy", "torch", "open3d", "MinkowskiEngine"]},
               "coarse_prediction_archive": str(prediction_archive), "coarse_prediction_sha256": coarse_hash,
               "refinement": args.icp, "icp_iteration_cap": args.icp_iterations,
               "icp_guard": "Keep initial pose unless full-source nearest-neighbor RMSE decreases and overlap within 0.08 does not decrease; no GT errors used",
               "convergence": "May stop earlier at Open3D default convergence tolerances",
               "thresholds": {"rre_deg": 5, "rte_normalized": .02},
               "metric": "Float64 symmetry-aware RRE using atan2; direct translation difference",
               "scope": "Full noiseless synthetic CAD pairs; correct pairing supplied. Not real-scan end-to-end validation.",
               "ground_truth_use": "Synthetic observation generation and scoring; never supplied to ICP",
               "output_directory": str(output)}
    for stage in ["before", "after"]:
        summary[stage] = {f"{key}_pass_percent": float(100 * frame[f"{stage}_{key}_pass"].mean())
                          for key in ["rotation", "translation", "joint"]}
        summary[stage]["median_rre_deg"] = float(frame[f"{stage}_rre_deg"].median())
        summary[stage]["median_rte_normalized"] = float(frame[f"{stage}_rte"].median())
    if hashlib.sha256(prediction_archive.read_bytes()).hexdigest() != coarse_hash:
        raise RuntimeError("Coarse prediction archive changed during refinement")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_comparison(frame, summary, output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

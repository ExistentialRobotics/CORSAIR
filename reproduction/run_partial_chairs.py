"""Fresh pilot on real partial Scan2CAD chairs, using upstream normalization."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from model import load_model, fc
from utils.Info.CADLib import CustomizeCADLib
from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.preprocess import apply_transform, load_raw_pc
from utils.read_json import to_T
from utils.symmetry import sym_pose
from utils.pose_metrics import pose_errors
from utils.pose_refinement import refine_pose_icp, guard_pose_update


def score(estimate, base_transform, symmetry):
    errors = []
    for k in range(symmetry):
        a = 2 * np.pi * k / symmetry
        c, s = np.cos(a), np.sin(a)
        sym = np.eye(4)
        sym[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        truth = np.linalg.inv(sym) @ np.linalg.inv(base_transform)
        errors.append(pose_errors(estimate, truth))
    return min(errors, key=lambda pair: pair[0])


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(output / 'per_chair.csv')
    arrays = np.load(output / 'poses_and_clouds.npz')
    ordered = frame[frame['mode'] == 'retrieved'].sort_values('after_rre_deg')
    examples = ordered.iloc[[0, len(ordered) // 2, len(ordered) - 1]]
    fig = plt.figure(figsize=(13, 12))
    for row_index, (_, row) in enumerate(examples.iterrows()):
        idx = int(row.query_index)
        scan = arrays[f'{idx}_observed'][::4]
        cad = arrays[f'{idx}_retrieved_cad'][::5]
        for column, stage in enumerate(['input', 'before', 'after']):
            ax = fig.add_subplot(3, 3, row_index * 3 + column + 1, projection='3d')
            xyz = scan if stage == 'input' else apply_transform(scan, arrays[f'{idx}_retrieved_{stage}'])
            if stage != 'input':
                ax.scatter(cad[:, 0], cad[:, 2], cad[:, 1], s=1, c='#bbbbbb', alpha=.3)
            ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], s=2,
                       c='#36878e' if stage == 'after' else '#bd7654', alpha=.65)
            title = f'Partial scan · query {idx}' if stage == 'input' else (
                f'{stage.capitalize()}: RRE {row[f"{stage}_rre_deg"]:.2f}°\n'
                f'RTE {row[f"{stage}_rte"]:.4f}')
            ax.set_title(title)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=45)
            ax.set_axis_off()
            if stage != 'input':
                ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2))
    iterations = json.loads((output / 'summary.json').read_text())['icp_iterations']
    fig.suptitle(f'Real partial Scan2CAD chairs · retrieved CAD + ICP (≤{iterations} steps)\n'
                 'Lowest, median, and highest final rotation error; gray = retrieved CAD', fontsize=15)
    fig.tight_layout(rect=(0, .02, 1, .94))
    fig.savefig(output / 'examples.png', dpi=170)
    plt.close(fig)


def register_pair(mode, idx, features, scan_xyz, target_features, target_xyz, target,
                  target_symmetry, truth_symmetry, observed, transform, iterations, seed, row):
    np.random.seed(seed)
    estimated, _, _, _, _ = sym_pose(features, scan_xyz, target_features, target_xyz,
                                     target_symmetry, seed=seed)
    before = estimated.numpy().astype(np.float64)
    proposal = refine_pose_icp(observed, target, before, max_iterations=iterations, guard=False)
    after, guard_details = guard_pose_update(observed, target, before, proposal)
    before_rre, before_rte = score(before, transform, truth_symmetry)
    proposal_rre, proposal_rte = score(proposal, transform, truth_symmetry)
    after_rre, after_rte = score(after, transform, truth_symmetry)
    row.update(before_rre_deg=before_rre, before_rte=before_rte,
               proposal_rre_deg=proposal_rre, proposal_rte=proposal_rte,
               after_rre_deg=after_rre, after_rte=after_rte,
               **{f'guard_{key}': value for key, value in guard_details.items()})
    return row, {f'{idx}_{mode}_before': before, f'{idx}_{mode}_after': after,
                 f'{idx}_{mode}_proposal': proposal, f'{idx}_{mode}_cad': target}


def summarize(frame, summary):
    summary['icp_guard'] = 'Accept lower full observed-scan nearest-neighbor RMSE with no loss of overlap at 0.08; otherwise keep the initial pose. No GT errors used.'
    for mode in ['retrieved', 'known_pair']:
        subset = frame[frame['mode'] == mode]
        summary[mode] = {'exact_id_percent': float(100 * subset.exact_id.mean())}
        if 'guard_accepted' in subset:
            summary[mode]['icp_accepted_count'] = int(subset.guard_accepted.sum())
        for stage in ['before', 'proposal', 'after']:
            if f'{stage}_rre_deg' not in subset:
                continue
            r, t = subset[f'{stage}_rre_deg'], subset[f'{stage}_rte']
            summary[mode][stage] = dict(rotation_pass_percent=float(100 * (r <= 5).mean()),
                translation_pass_percent=float(100 * (t <= .02).mean()),
                joint_pass_percent=float(100 * ((r <= 5) & (t <= .02)).mean()),
                median_rre_deg=float(r.median()), median_rte=float(t.median()))
    return summary


def apply_guard_to_results(output):
    """Apply the label-free guard to this pilot's saved fresh ICP proposals."""
    output = Path(output)
    frame = pd.read_csv(output / 'per_chair.csv')
    with np.load(output / 'poses_and_clouds.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    labels = {Path(line.split()[0]).stem: int(line.split()[1]) for line in
              (ROOT / 'configs/03001627_scan2cad_rot_sym_label.txt').read_text().splitlines()}
    for row_idx, row in frame.iterrows():
        idx, mode = int(row.query_index), row['mode']
        prefix = f'{idx}_{mode}'
        proposal_key = f'{prefix}_proposal'
        if proposal_key not in arrays:
            arrays[proposal_key] = arrays[f'{prefix}_after'].copy()
        proposal = arrays[proposal_key]
        selected, details = guard_pose_update(arrays[f'{idx}_observed'], arrays[f'{prefix}_cad'],
                                              arrays[f'{prefix}_before'], proposal)
        arrays[f'{prefix}_after'] = selected
        for stage, transform in [('proposal', proposal), ('after', selected)]:
            rre, rte = score(transform, arrays[f'{idx}_transform'], labels[row.gt_id])
            frame.loc[row_idx, f'{stage}_rre_deg'] = rre
            frame.loc[row_idx, f'{stage}_rte'] = rte
        for key, value in details.items():
            frame.loc[row_idx, f'guard_{key}'] = value
    np.savez_compressed(output / 'poses_and_clouds.npz', **arrays)
    frame.to_csv(output / 'per_chair.csv', index=False)
    summary = summarize(frame, json.loads((output / 'summary.json').read_text()))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_results(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'partial-chairs')
    parser.add_argument('--models', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--icp-iterations', type=int, default=10)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    started = time.monotonic()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    cadroot = str(args.data / 'ShapeNetCore.v2.PC15k')
    scanroot = str(args.data / 'Scan2CAD_pc')
    info = Scan2cadInfo(cadroot, scanroot, '03001627', str(args.data / 'Scan2CAD_annotations'))
    library = CustomizeCADLib(cadroot, '03001627', info.UsedObjId,
                             str(ROOT / 'configs/03001627_scan2cad.npy'), .03, preload=False)
    files, matches, cadposes, scanposes = info.get_split('test')
    if not 0 < args.models <= len(files):
        raise ValueError('models must be between 1 and the held-out query count')
    selected = np.sort(np.random.RandomState(args.seed).choice(len(files), args.models, replace=False))
    (output / 'selection.json').write_text(json.dumps({
        'seed': args.seed, 'indices': selected.tolist(), 'files': [files[i] for i in selected],
        'total_test_queries': len(files), 'library_size': len(library)}, indent=2))
    model = load_model('ResUNetBN2C')(in_channels=1, out_channels=16, bn_momentum=.05,
                                     normalize_feature=True, conv1_kernel_size=3, D=3).eval()
    embedding = fc.conv1_max_embedding(1024, 512, 256).eval()
    checkpoint_path = ROOT / 'ckpts/scannet_ret_chair'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    embedding.load_state_dict(checkpoint['embedding_state_dict'])

    def extract(items):
        data = library.collate_pair_fn(items)
        with torch.no_grad():
            sparse = ME.SparseTensor(data['base_feat'], data['base_coords'])
            local, global_features = model(sparse)
            descriptors = torch.nn.functional.normalize(embedding(global_features), dim=1).numpy()
        result = []
        for j in range(len(items)):
            mask = local.C[:, 0] == j
            result.append((local.F[mask].numpy(), data['base_origin'][mask].numpy(), descriptors[j]))
        return result

    cad_features = []
    for start in range(0, len(library), 16):
        cad_features.extend(extract([library[i] for i in range(start, min(start + 16, len(library)))]))
        print(f'CAD descriptors {len(cad_features)}/{len(library)}', flush=True)
    descriptors = np.stack([x[2] for x in cad_features])
    fixed = np.load(ROOT / 'configs/fix_trans.npy')
    symmetries = [int(line.split()[1]) for line in
                  (ROOT / 'configs/03001627_scan2cad_rot_sym_label.txt').read_text().splitlines()]
    records, arrays, tasks = [], {}, []
    for count, idx in enumerate(selected):
        gt_idx = library.id2idx[matches[idx]]
        scan = load_raw_pc(os.path.join(scanroot, files[idx]), 10000)
        gt_cad = load_raw_pc(library.CadPcs[gt_idx], 10000)
        scan_pose = to_T(scanposes[idx]['translation'], scanposes[idx]['rotation'], scanposes[idx]['scale'])
        cad_pose = to_T(cadposes[idx]['translation'], cadposes[idx]['rotation'], cadposes[idx]['scale'])
        center = gt_cad.mean(0)
        radius = np.max(np.linalg.norm(gt_cad - center, axis=1))
        canonical_scan = (apply_transform(scan, np.linalg.inv(cad_pose) @ scan_pose) - center) / radius
        transform = fixed[idx, 0]
        observed = np.ascontiguousarray(apply_transform(canonical_scan, transform))
        quantized, grid, _ = library.quant(observed, canonical_scan)
        scan_features, scan_xyz, descriptor = extract([dict(coord=grid, origin=quantized,
            feat=np.ones((len(grid), 1)), idx=int(idx), T=transform)])[0]
        retrieved_idx = int(np.argmax(descriptors @ descriptor))
        arrays[f'{idx}_observed'] = observed
        arrays[f'{idx}_transform'] = transform
        for mode, target_idx in [('retrieved', retrieved_idx), ('known_pair', gt_idx)]:
            features, xyz, _ = cad_features[target_idx]
            pair_seed = args.seed + int(idx)
            target = library._getpc(target_idx)
            row = dict(query_index=int(idx), file=files[idx], mode=mode, gt_id=matches[idx],
                       retrieved_id=library.ids[target_idx], exact_id=target_idx == gt_idx,
                       scan_points=len(observed), voxel_points=len(scan_xyz))
            tasks.append(delayed(register_pair)(mode, int(idx), scan_features, scan_xyz, features, xyz,
                target, symmetries[target_idx], symmetries[gt_idx], observed, transform,
                args.icp_iterations, pair_seed, row))
        print(f'Prepared partial query {count+1}/{len(selected)}', flush=True)
    completed = Parallel(n_jobs=args.workers, backend='loky', return_as='generator_unordered')(tasks)
    for row, pair_arrays in completed:
        records.append(row)
        arrays.update(pair_arrays)
        pd.DataFrame(records).sort_values(['query_index', 'mode']).to_csv(output / 'per_chair.csv', index=False)
        print(f'{len(records)}/{len(tasks)} query {row["query_index"]} {row["mode"]}: '
              f'RRE {row["before_rre_deg"]:.3f} -> {row["after_rre_deg"]:.3f}; '
              f'RTE {row["before_rte"]:.5f} -> {row["after_rte"]:.5f}', flush=True)
        np.savez_compressed(output / 'poses_and_clouds.npz', **arrays)
    frame = pd.DataFrame(records)
    summary = dict(models=args.models, seed=args.seed, library_size=len(library),
                   test_queries=len(files), device='cpu', icp_iterations=args.icp_iterations,
                   checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                   elapsed_seconds=time.monotonic() - started,
                   protocol='Real segmented partial Scan2CAD scans; random held-out query subset. '
                   'Top-1 learned CAD retrieval plus a separate known-pair diagnostic. '
                   'Upstream GT CAD pose/scale normalization and fixed test transforms retained. '
                   'RTE is in unit-radius-normalized coordinates. GT symmetry used for scoring. '
                   'No prediction caches; standard point-to-point ICP scan-to-CAD, distance 0.08.')
    summary = summarize(frame, summary)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_results(output)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

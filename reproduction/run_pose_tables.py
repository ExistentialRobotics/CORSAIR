"""Separate known-pair ShapeNet and partial Scan2CAD registration evaluations."""
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
from scipy.spatial import cKDTree
from transforms3d.euler import euler2mat
from model import load_model
from utils.Info.CADLib import CustomizeCADLib
from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.preprocess import apply_transform, load_raw_pc, load_norm_pc
from utils.read_json import to_T
from utils.symmetry import sym_pose
from utils.pose_metrics import pose_errors
from utils.pose_refinement import refine_pose_icp, guard_pose_update


def symmetry_matrices(order):
    result = []
    for k in range(order):
        transform = np.eye(4)
        transform[:3, :3] = euler2mat(0, 2 * np.pi * k / order, 0)
        result.append(transform)
    return result


def infer_symmetry(points):
    tree = cKDTree(points)
    for order in [12, 8, 6, 4, 3, 2, 1]:
        good = True
        for symmetry in symmetry_matrices(order)[1:order // 2 + 1]:
            rotated = apply_transform(points, symmetry)
            distance = max(tree.query(rotated)[0].max(), cKDTree(rotated).query(points)[0].max())
            if distance > .1:
                good = False
                break
        if good:
            return order
    return 1


def quantize(points):
    points = np.ascontiguousarray(points)
    grid = np.ascontiguousarray(np.floor(points / .03))
    indices = ME.utils.sparse_quantize(grid, return_index=True, return_maps_only=True)
    return points[indices], grid[indices]


def evaluate_case(case, iterations):
    torch.set_num_threads(1)
    np.random.seed(case['registration_seed'])
    source, target = case['source'], case['target']
    arrays = dict(source=source, target=target, truth_variants=case['truth_variants'])
    error = ''
    try:
        sym, _, ransac, _, _ = sym_pose(case['source_features'], case['source_quantized'],
            case['target_features'], case['target_quantized'], case['symmetry_order'],
            max_corr=case['max_corr'], seed=case['registration_seed'])
        initial = sym.numpy().astype(np.float64)
        ransac = ransac.numpy().astype(np.float64)
        proposal = refine_pose_icp(source, target, initial, max_iterations=iterations, guard=False)
        refined, guard = guard_pose_update(source, target, initial, proposal)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        initial = ransac = proposal = refined = np.full((4, 4), np.nan)
        guard = dict(accepted=False)
    arrays.update(ransac=ransac, corsair=initial, proposal=proposal, refined=refined)
    rows = []
    for method, estimate in [('RANSAC', ransac), ('CORSAIR', initial), ('CORSAIR + ICP', refined)]:
        rre, rte = (float('inf'), float('inf')) if error else min(
            (pose_errors(estimate, truth) for truth in case['truth_variants']), key=lambda pair: pair[0])
        rows.append(dict(table=case['table'], category=case['category'], case_index=case['case_index'],
            query_id=case['query_id'], cad_id=case['cad_id'], method=method,
            symmetry_order=case['symmetry_order'], rre_deg=rre, rte=rte, error=error,
            guard_accepted=guard['accepted'] if method == 'CORSAIR + ICP' else False))
    return rows, arrays, guard


def write_tables(records, output):
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'per_case.csv', index=False)
    rendered = []
    for number in [1, 3]:
        rows = []
        for (category, method), group in frame[frame.table == number].groupby(['category', 'method'], sort=False):
            row = dict(Category=category.title(), Method=method, N=len(group))
            if number == 3:
                row['With Sym'] = 'No' if method == 'RANSAC' else 'Yes'
            for threshold in ([5, 15] if number == 1 else [5, 15, 45]):
                row[f'RRE≤{threshold}°'] = round(float(100 * (group.rre_deg <= threshold).mean()), 2)
            for threshold in ([.02, .05] if number == 1 else [.02, .05, .1]):
                row[f'RTE≤{threshold:.2f}'] = round(float(100 * (group.rte <= threshold).mean()), 2)
            row['Both ≤5°/0.02'] = round(float(100 * ((group.rre_deg <= 5) & (group.rte <= .02)).mean()), 2)
            rows.append(row)
        if not rows:
            continue
        table = pd.DataFrame(rows)
        table['order'] = table.Method.map({'RANSAC': 0, 'CORSAIR': 1, 'CORSAIR + ICP': 2})
        table = table.sort_values(['Category', 'order']).drop(columns='order')
        table.to_csv(output / f'table{number}.csv', index=False)
        title = 'ShapeNet: full noiseless CAD pairs' if number == 1 else 'Scan2CAD: real partial object scans'
        rendered.append(f'<h2>Table {number}. {title}</h2>{table.to_html(index=False)}')
    (output / 'tables.html').write_text('<!doctype html><meta charset="utf-8"><title>CORSAIR pose evaluation</title>'
        '<style>body{font:16px system-ui;margin:40px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}</style>'
        '<h1>Pose registration evaluation</h1><p>All entries are percentages. '
        'Correct CAD pairing supplied. RTE uses unit-radius-normalized coordinates. '
        'ICP has at most 10 iterations; updates use geometric distance and overlap, without a GT acceptance gate. '
        'Joint success requires both thresholds on the same case.</p>' + ''.join(rendered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=int, default=100)
    parser.add_argument('--tables', type=int, nargs='+', choices=[1, 3], default=[1, 3])
    parser.add_argument('--categories', nargs='+', choices=['chair', 'table'], default=['chair', 'table'])
    parser.add_argument('--workers', type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pose-tables')
    parser.add_argument('--icp-iterations', type=int, default=10)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    output = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    cases, manifest = [], dict(models_per_category=args.models, workers=args.workers,
        icp_iterations=args.icp_iterations, pairing='ground truth CAD, both protocols',
        table1_seed=0, table3_subset_seed=42, ransac_n=10, k_nn=5, voxel_size=.03,
        table1_max_corr=.4, table3_max_corr=.2, checkpoints={}, selections={},
        preprocessing='Unit-radius CAD normalization; Scan2CAD additionally uses GT CAD pose/scale, GT segmentation and upstream fixed test transforms.')
    for category, catid in [('chair', '03001627'), ('table', '04379243')]:
        if category not in args.categories:
            continue
        ckpt = ROOT / f'ckpts/scannet_pose_{category}_best'
        state = torch.load(ckpt, map_location='cpu')
        model = load_model('ResUNetBN2C')(in_channels=1, out_channels=16, bn_momentum=.05,
            normalize_feature=True, conv1_kernel_size=3, D=3).eval()
        model.load_state_dict(state['state_dict'])
        manifest['checkpoints'][category] = dict(path=ckpt.name, sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())

        def extract(points):
            xyz, grid = quantize(points)
            coords, feats = ME.utils.sparse_collate([torch.from_numpy(grid)], [torch.ones((len(grid), 1))])
            with torch.no_grad():
                local, _ = model(ME.SparseTensor(feats.float(), coords.int()))
            return xyz.astype(np.float32), local.F.numpy()

        def prepare(number, index, query_id, cad_id, source, target, truths, symmetry, seed):
            source_xyz, source_feat = extract(source)
            target_xyz, target_feat = extract(target)
            cases.append(dict(table=number, category=category, case_index=index, query_id=query_id,
                cad_id=cad_id, source=source, target=target, truth_variants=truths,
                symmetry_order=symmetry, registration_seed=seed, max_corr=.4 if number == 1 else .2,
                source_quantized=source_xyz, source_features=source_feat,
                target_quantized=target_xyz, target_features=target_feat))

        files = sorted((ROOT / f'docker/data/ShapeNetCore.v2.PC15k/{catid}/test').glob('*.npy')) if 1 in args.tables else []
        if 1 in args.tables and not 0 < args.models <= len(files):
            raise ValueError('Requested model count exceeds the bundled ShapeNet subset')
        if args.models < len(files):
            files = sorted(np.random.RandomState(0).choice(files, args.models, replace=False))
        rng = np.random.RandomState(0)
        for index, path in enumerate(files):
            source = np.load(path)
            source = source - source.mean(0)
            source = source / np.linalg.norm(source, axis=1).max()
            truth = np.eye(4)
            truth[:3, :3] = euler2mat(*np.deg2rad(rng.uniform(-360, 360, 3)))
            truth[:3, 3] = rng.uniform(-1, 1, 3)
            target = apply_transform(source, truth)
            symmetry = infer_symmetry(source)
            truths = np.stack([truth @ sym for sym in symmetry_matrices(symmetry)])
            prepare(1, index, path.name, path.stem, source, target, truths, symmetry, index)
            print(f'Prepared Table 1 {category} {index+1}/{len(files)}', flush=True)

        manifest['selections'][category] = dict(shapenet_files=[p.name for p in files])
        if 3 not in args.tables:
            continue
        cadroot, scanroot = str(args.data / 'ShapeNetCore.v2.PC15k'), str(args.data / 'Scan2CAD_pc')
        info = Scan2cadInfo(cadroot, scanroot, catid, str(args.data / 'Scan2CAD_annotations'))
        library = CustomizeCADLib(cadroot, catid, info.UsedObjId,
            str(ROOT / f'configs/{catid}_scan2cad.npy'), .03, preload=False)
        scan_files, matches, cadposes, scanposes = info.get_split('test')
        selected = np.sort(np.random.RandomState(42).choice(len(scan_files), args.models, replace=False))
        manifest['selections'][category].update(
            scan2cad_indices=selected.tolist(), scan2cad_files=[scan_files[i] for i in selected],
            available_scan2cad_queries=len(scan_files))
        labels = [int(line.split()[1]) for line in (ROOT / f'configs/{catid}_scan2cad_rot_sym_label.txt').read_text().splitlines()]
        fixed = np.load(ROOT / 'configs/fix_trans.npy')
        for count, idx in enumerate(selected):
            cad_idx = library.id2idx[matches[idx]]
            cad = load_raw_pc(library.CadPcs[cad_idx], 10000)
            scan = load_raw_pc(os.path.join(scanroot, scan_files[idx]), 10000)
            scan_pose = to_T(scanposes[idx]['translation'], scanposes[idx]['rotation'], scanposes[idx]['scale'])
            cad_pose = to_T(cadposes[idx]['translation'], cadposes[idx]['rotation'], cadposes[idx]['scale'])
            center = cad.mean(0)
            radius = np.linalg.norm(cad - center, axis=1).max()
            canonical_scan = (apply_transform(scan, np.linalg.inv(cad_pose) @ scan_pose) - center) / radius
            source = apply_transform(canonical_scan, fixed[idx, 0])
            target = (cad - center) / radius
            symmetry = labels[cad_idx]
            truths = np.stack([sym @ np.linalg.inv(fixed[idx, 0]) for sym in symmetry_matrices(symmetry)])
            prepare(3, int(idx), scan_files[idx], matches[idx], source, target, truths, symmetry, 42 + int(idx))
            print(f'Prepared Table 3 {category} {count+1}/{args.models}', flush=True)
    (output / 'protocol.json').write_text(json.dumps(manifest, indent=2))
    print(f'Registering {len(cases)} fresh pairs with {args.workers} CPU workers. Output: {output}', flush=True)
    records = []
    completed = Parallel(n_jobs=args.workers, backend='loky', return_as='generator_unordered')(
        delayed(evaluate_case)(case, args.icp_iterations) for case in cases)
    for count, (rows, arrays, guard) in enumerate(completed, 1):
        row = rows[0]
        folder = output / f'table{row["table"]}' / row['category']
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f'{row["case_index"]:04d}.npz', **arrays)
        (folder / f'{row["case_index"]:04d}.json').write_text(json.dumps(dict(rows=rows, guard=guard), indent=2))
        records.extend(rows)
        write_tables(records, output)
        print(f'Completed {count}/{len(cases)}: Table {row["table"]} {row["category"]} {row["query_id"]}'
              f' RRE {rows[1]["rre_deg"]:.3f} -> {rows[2]["rre_deg"]:.3f},'
              f' RTE {rows[1]["rte"]:.5f} -> {rows[2]["rte"]:.5f}', flush=True)
    manifest['elapsed_seconds'] = time.monotonic() - started
    manifest['completed_pairs'] = len(records) // 3
    (output / 'protocol.json').write_text(json.dumps(manifest, indent=2))
    if set(args.categories) == {'chair', 'table'}:
        from reproduction.report_pose_tables import collect
        collect([output], output, args.models)
    print(f'Finished: {output}', flush=True)


if __name__ == '__main__':
    main()

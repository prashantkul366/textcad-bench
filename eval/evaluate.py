"""
TextCAD-Bench unified evaluator.

Usage:
    python eval/evaluate.py \
        --results_file results/my_model_t1_results.json \
        --gt_dir data/gt_meshes \
        --output results/my_model_t1_metrics.json

The results_file should be a JSON list of dicts with at least:
    uid            : sample UID (e.g. "0062/00625492")
    pred_mesh_path : path to predicted OBJ/STL (or None if failed)
    generated_code : (optional) generated code string

If pred_mesh_path is not present, set ir_compile=1 for that sample.
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from metrics import evaluate_pair, load_and_normalize


FAILED_UIDS = {
    '0056/00564141', '0093/00939079', '0072/00721093', '0031/00316178',
    '0064/00644281', '0031/00319711', '0071/00711660', '0042/00421905',
    '0028/00281866', '0064/00649509',
}


def uid_to_filename(uid):
    """Convert UID like '0062/00625492' to filename '00625492.obj'."""
    return uid.split('/')[-1] + '.obj'


def summarise(results):
    total  = [r for r in results if not r.get('skip')]
    valid  = [r for r in total  if r['ir_compile'] == 0]
    cds    = [r['cd_mean']   for r in valid if r['cd_mean']   is not None]
    ious   = [r['iou']       for r in valid if r['iou']       is not None]
    wt     = [r for r in valid if r.get('watertight')]

    summary = {
        'n_total':    len(total),
        'n_valid':    len(valid),
        'n_skipped':  len(results) - len(total),
        'ir_pct':     round((1 - len(valid) / len(total)) * 100, 2) if total else 0,
        'cd_mean':    round(float(np.mean(cds)),   2) if cds  else None,
        'cd_median':  round(float(np.median(cds)), 2) if cds  else None,
        'iou_mean':   round(float(np.mean(ious)),  2) if ious else None,
        'wt_pct':     round(len(wt) / len(valid) * 100, 1) if valid else 0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', required=True,
                        help='JSON file with model outputs')
    parser.add_argument('--gt_dir', required=True,
                        help='Directory containing GT OBJ meshes')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: results_file + _eval.json)')
    parser.add_argument('--uid_field', default='uid',
                        help='Field name for sample UID')
    parser.add_argument('--mesh_field', default='pred_mesh_path',
                        help='Field name for predicted mesh path')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_file.replace('.json', '_eval.json')

    data    = json.load(open(args.results_file))
    results = []

    for item in tqdm(data, desc='Evaluating'):
        uid = item.get(args.uid_field, '')

        if uid in FAILED_UIDS:
            results.append({**item, 'skip': True,
                             'ir_compile': 1, 'cd_mean': None, 'iou': None})
            continue

        gt_path   = os.path.join(args.gt_dir, uid_to_filename(uid))
        pred_path = item.get(args.mesh_field)

        metrics = evaluate_pair(pred_path, gt_path)
        results.append({**item, **metrics, 'skip': False})

    summary = summarise(results)

    output = {'summary': summary, 'results': results}
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Results: {args.results_file} ===")
    print(f"  n={summary['n_total']}  IR={summary['ir_pct']}%  "
          f"CD={summary['cd_mean']}  IoU={summary['iou_mean']}  "
          f"WT={summary['wt_pct']}%")
    print(f"  Saved to: {args.output}")


if __name__ == '__main__':
    main()

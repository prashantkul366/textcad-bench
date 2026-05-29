"""
Batch evaluation across all TextCAD-Bench result files.

Usage:
    python eval/evaluate_all.py \
        --results_dir results/ \
        --gt_dir data/ \
        --output summary.json
"""
import argparse, json, os, sys, numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import load_and_normalize, chamfer_distance, voxel_iou, evaluate_pair

RESULT_FILES = {
    'qwen7b_t1':   ('qwen7b_t1_results.json',   'gt_meshes',           'T1'),
    'qwen7b_t2':   ('qwen7b_t2_results.json',   'gt_meshes_t2',        'T2'),
    'qwen7b_t3':   ('qwen7b_t3_results.json',   'gt_meshes_cadprompt', 'T3'),
    'llama8b_t1':  ('llama8b_t1_results.json',  'gt_meshes',           'T1'),
    'llama8b_t2':  ('llama8b_t2_results.json',  'gt_meshes_t2',        'T2'),
    'llama8b_t3':  ('llama8b_t3_results.json',  'gt_meshes_cadprompt', 'T3'),
    'text2cad_t1': ('text2cad_t1_results.json', 'gt_meshes',           'T1'),
    'text2cad_t2': ('text2cad_t2_results.json', 'gt_meshes_t2',        'T2'),
    'text2cad_t3': ('text2cad_t3_results.json', 'gt_meshes_cadprompt', 'T3'),
    'cadmium_t1':  ('cadmium_t1_results.json',  'gt_meshes',           'T1'),
    'cadmium_t2':  ('cadmium_t2_results.json',  'gt_meshes_t2',        'T2'),
    't2cq_t1':     ('t2cq_t1_results.json',     'gt_meshes',           'T1'),
    't2cq_t2':     ('t2cq_t2_results.json',     'gt_meshes_t2',        'T2'),
    't2cq_t3':     ('t2cq_t3_results.json',     'gt_meshes_cadprompt', 'T3'),
    'cadrille_t1': ('cadrille_t1_results.json', 'gt_meshes',           'T1'),
    'cadrille_t2': ('cadrille_t2_results.json', 'gt_meshes_t2',        'T2'),
    'cadrille_t3': ('cadrille_t3_results.json', 'gt_meshes_cadprompt', 'T3'),
    'procad_t1':   ('procad_t1_results.json',   'gt_meshes',           'T1'),
    'procad_t2':   ('procad_t2_results.json',   'gt_meshes_t2',        'T2'),
    'procad_t3':   ('procad_t3_results.json',   'gt_meshes_cadprompt', 'T3'),
}

def agg(data):
    total = [r for r in data if not r.get('skip')]
    valid = [r for r in total if r['ir_compile']==0]
    cds   = [r['cd_mean'] for r in valid if r['cd_mean']]
    ious  = [r['iou']     for r in valid if r['iou']]
    if not total: return None
    return {
        'n':    len(total),
        'ir':   round((1-len(valid)/len(total))*100, 2),
        'cd':   round(float(np.mean(cds)),  2) if cds  else None,
        'iou':  round(float(np.mean(ious)), 2) if ious else None,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--gt_dir',      required=True)
    parser.add_argument('--output',      default='summary.json')
    args = parser.parse_args()

    summary = {}
    print(f"\n{'Key':<15} {'n':>5} {'IR%':>6} {'CD':>8} {'IoU':>7}")
    print("="*45)

    for key, (fname, gt_subdir, ts) in RESULT_FILES.items():
        path = os.path.join(args.results_dir, fname)
        if not os.path.exists(path):
            print(f"{key:<15} {'MISSING':>5}")
            continue
        data = json.load(open(path))
        s    = agg(data)
        if s:
            summary[key] = s
            cd_s  = f"{s['cd']:.2f}"  if s['cd']  else "N/A"
            iou_s = f"{s['iou']:.2f}" if s['iou'] else "N/A"
            print(f"{key:<15} {s['n']:>5} {s['ir']:>6.1f} {cd_s:>8} {iou_s:>7}")

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved to {args.output}")

if __name__=='__main__': main()

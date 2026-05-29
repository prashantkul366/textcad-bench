"""
Reproduce all figures from TextCAD-Bench paper.
Usage:
    python scripts/generate_figures.py --results_dir results/ --out_dir figs/
Figures generated:
    fig_rank_reversal.pdf      -- CD rank trajectories across T1/T2/T3
    fig_ir_cd_scatter.pdf      -- IR vs CD bubble plot (GoR as bubble size)
    fig_primitives_ir.pdf      -- IR by complexity bin
    fig_failure_distribution.pdf -- Per-model failure mode breakdown
    fig_annotation_divergence.pdf -- Cosine distance histogram
    fig_dataset_analysis.pdf   -- n_ops vs n_primitives distributions
"""
import argparse
import json
from pathlib import Path
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--out_dir",     default="figs/")
    return p.parse_args()
def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[generate_figures] Reading results from: {args.results_dir}")
    print(f"[generate_figures] Writing figures  to:  {args.out_dir}")
    # TODO: implement individual figure functions
    # Each function should load args.results_dir/model_results.json
    # and write a PDF to args.out_dir
    raise NotImplementedError(
        "Figure generation scripts are under active development. "
        "See paper supplementary for manual reproduction steps."
    )
if __name__ == "__main__":
    main()

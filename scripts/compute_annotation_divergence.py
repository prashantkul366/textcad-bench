"""
Compute annotation divergence between Text2CAD v1.0 and v1.1.
Reproduces the 0.377 cosine distance result from the paper.

Usage:
    python scripts/compute_annotation_divergence.py \
        --v10 data/text2cad_v1.0.csv \
        --v11 data/text2cad_v1.1.csv \
        --n_samples 1000 \
        --output figs/annotation_divergence.pdf
"""

import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v10',       required=True, help='Text2CAD v1.0 CSV')
    parser.add_argument('--v11',       required=True, help='Text2CAD v1.1 CSV')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--output',    default='figs/annotation_divergence.pdf')
    args = parser.parse_args()

    df10 = pd.read_csv(args.v10)
    df11 = pd.read_csv(args.v11)

    overlap = set(df10['uid']) & set(df11['uid'])
    print(f"Overlapping UIDs: {len(overlap):,}")

    random.seed(args.seed)
    sample_uids = random.sample(sorted(overlap),
                                min(args.n_samples, len(overlap)))

    idx10 = df10.set_index('uid')
    idx11 = df11.set_index('uid')

    texts_v10, texts_v11 = [], []
    for uid in sample_uids:
        texts_v10.append(str(idx10.loc[uid, 'abstract']))
        texts_v11.append(str(idx11.loc[uid, 'abstract']))

    print(f"Embedding {len(texts_v10)} pairs with all-mpnet-base-v2...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    emb10 = model.encode(texts_v10, batch_size=64, show_progress_bar=True)
    emb11 = model.encode(texts_v11, batch_size=64, show_progress_bar=True)

    sims  = [cosine_similarity(emb10[i:i+1], emb11[i:i+1])[0][0]
             for i in range(len(emb10))]
    dists = [1 - s for s in sims]

    print(f"\nCosine distance stats (n={len(dists)}):")
    print(f"  mean:   {np.mean(dists):.4f}")
    print(f"  median: {np.median(dists):.4f}")
    print(f"  std:    {np.std(dists):.4f}")
    print(f"  p25:    {np.percentile(dists, 25):.4f}")
    print(f"  p75:    {np.percentile(dists, 75):.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.hist(dists, bins=40, color='#7F77DD', alpha=0.85,
            edgecolor='white', linewidth=0.3)
    ax.axvline(np.mean(dists), color='#D85A30', linewidth=1.5,
               linestyle='--', label=f'Mean={np.mean(dists):.3f}')
    ax.axvline(np.median(dists), color='#1D9E75', linewidth=1.5,
               linestyle=':', label=f'Median={np.median(dists):.3f}')
    ax.set_xlabel('Cosine distance (v1.0 vs v1.1 annotations)')
    ax.set_ylabel('Count')
    ax.set_title(f'Annotation divergence (n={len(dists):,})')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Figure saved to {args.output}")


if __name__ == '__main__':
    main()

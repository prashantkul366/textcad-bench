# TextCAD-Bench

**A Unified Benchmark for Evaluating LLM-Based Text-to-CAD Generation**

Paper: *TextCAD-Bench: A Unified Benchmark for Evaluating LLM-Based Text-to-CAD Generation* (INCECT 2026)

---

## Overview

TextCAD-Bench provides three frozen test sets, unified evaluation metrics, and inference scripts for seven models across three paradigms. Our main finding: **model rankings completely reverse across test sets** — no model is consistently best across all three benchmarks.

| Test Set | Source | Samples | Description |
|----------|--------|---------|-------------|
| T1 | DeepCAD / Text2CAD | 2,000 | Stratified by complexity (n_primitives) |
| T2 | Fusion360 / CADmium | 1,725 | Real engineering parts, GPT-4.1 annotated |
| T3 | CADPrompt | 200 | Human-authored CadQuery-style prompts |

### Key Results

| Model | Paradigm | T1 CD↓ | T2 CD↓ | T3 CD↓ |
|-------|----------|--------|--------|--------|
| Text-to-CadQuery | C (SFT) | **17.30** | 22.09 | 26.98 |
| Llama-3-8B | D (zero-shot) | 18.07 | **22.33** | 27.11 |
| Qwen-7B | D (zero-shot) | 19.85 | 24.62 | **26.82** |
| CADmium-7B | B (token) | 21.75 | **16.75** | N/A |
| ProCAD | C (SFT) | 22.88 | 22.54 | 28.41 |
| Text2CAD | B (token) | 26.71 | 23.26 | 34.13 |
| cadrille | C (RL) | 42.67 | 43.35 | 42.60 |

CD ×10², lower is better. No model is #1 across all three sets.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/textcad-bench.git
cd textcad-bench
pip install -r requirements.txt
```

---

## Dataset

Download from HuggingFace (replace with your repo):

```bash
huggingface-cli download YOUR_HF_REPO \
    t1_samples.json t2_samples.json t3_cadprompt.json \
    gt_meshes.zip gt_meshes_t2.zip gt_meshes_cadprompt.zip \
    --local-dir ./data
cd data && unzip gt_meshes.zip && unzip gt_meshes_t2.zip && unzip gt_meshes_cadprompt.zip
```

T1 MD5: `15aa091ffd6d8f5a7c50b7a49af9ef6c`

---

## Evaluation Protocol

- **CD**: meshes normalised to [-0.5, 0.5]³, 8,192 surface points, mean bidirectional NN distance ×10²
- **IR**: fraction of outputs failing compile or producing empty mesh  
- **IoU**: 64³ voxelisation, watertight meshes only

```bash
python eval/evaluate.py --results_file results/your_results.json \
    --gt_dir data/gt_meshes --output results/metrics.json
```

---

## Running Inference

```bash
# Text-to-CadQuery
python models/run_t2cq.py --test_set T1 \
    --data data/t1_samples.json --gt_dir data/gt_meshes \
    --output results/t2cq_t1_results.json

# Qwen-7B zero-shot
python models/run_qwen7b.py --test_set T1 \
    --data data/t1_samples.json --gt_dir data/gt_meshes \
    --output results/qwen7b_t1_results.json

# All models, all test sets
bash scripts/run_all_models.sh
```

---

## Repository Structure

```
textcad-bench/
  eval/
    evaluate.py          # Core evaluation: CD, IR, IoU
    evaluate_all.py      # Batch evaluation
    metrics.py           # Metric implementations
    mesh_utils.py        # Mesh loading and normalisation
  models/
    run_qwen7b.py        # Qwen-7B zero-shot
    run_llama8b.py       # Llama-3-8B zero-shot
    run_text2cad.py      # Text2CAD
    run_cadmium.py       # CADmium-7B
    run_t2cq.py          # Text-to-CadQuery
    run_cadrille.py      # cadrille
    run_procad.py        # ProCAD
    cadquery_executor.py # Shared CadQuery execution
    cadmium_converter.py # CADmium JSON → mesh
  scripts/
    run_all_models.sh    # Run all inference
    generate_figures.py  # Reproduce paper figures
    compute_annotation_divergence.py
  data/                  # Download from HF
  results/               # Model outputs
  requirements.txt
  README.md
```

---

## Citation

```bibtex
booktitle = {Proceedings of INCECT},
year      = {2026},
note      = {Under review}
```

## License

MIT License.

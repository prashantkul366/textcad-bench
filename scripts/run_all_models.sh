#!/bin/bash
# Run all TextCAD-Bench models across T1, T2, T3.
# Edit DATA_DIR and GT_DIR to match your setup.
# Each model needs a separate GPU session (~3-12 hours each).

set -e

DATA_DIR="./data"
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

echo "=== TextCAD-Bench: Full Evaluation ==="
echo "Results will be saved to: $RESULTS_DIR"

# ── Group D: Zero-shot LLMs ────────────────────────────────────────────────────

echo "[1/7] Qwen-7B T1..."
python models/run_qwen7b.py --test_set T1 \
    --data "$DATA_DIR/t1_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes" \
    --output "$RESULTS_DIR/qwen7b_t1_results.json"

echo "[1/7] Qwen-7B T2..."
python models/run_qwen7b.py --test_set T2 \
    --data "$DATA_DIR/t2_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes_t2" \
    --output "$RESULTS_DIR/qwen7b_t2_results.json"

echo "[1/7] Qwen-7B T3..."
python models/run_qwen7b.py --test_set T3 \
    --data "$DATA_DIR/t3_cadprompt.json" \
    --gt_dir "$DATA_DIR/gt_meshes_cadprompt" \
    --output "$RESULTS_DIR/qwen7b_t3_results.json"

echo "[2/7] Llama-3-8B T1..."
python models/run_llama8b.py --test_set T1 \
    --data "$DATA_DIR/t1_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes" \
    --output "$RESULTS_DIR/llama8b_t1_results.json"

echo "[2/7] Llama-3-8B T2..."
python models/run_llama8b.py --test_set T2 \
    --data "$DATA_DIR/t2_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes_t2" \
    --output "$RESULTS_DIR/llama8b_t2_results.json"

echo "[2/7] Llama-3-8B T3..."
python models/run_llama8b.py --test_set T3 \
    --data "$DATA_DIR/t3_cadprompt.json" \
    --gt_dir "$DATA_DIR/gt_meshes_cadprompt" \
    --output "$RESULTS_DIR/llama8b_t3_results.json"

# ── Group B: Token-based ───────────────────────────────────────────────────────

echo "[3/7] Text2CAD T1..."
python models/run_text2cad.py --test_set T1 \
    --data "$DATA_DIR/t1_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes" \
    --output "$RESULTS_DIR/text2cad_t1_results.json"

echo "[3/7] Text2CAD T2..."
python models/run_text2cad.py --test_set T2 \
    --data "$DATA_DIR/t2_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes_t2" \
    --output "$RESULTS_DIR/text2cad_t2_results.json"

echo "[3/7] Text2CAD T3..."
python models/run_text2cad.py --test_set T3 \
    --data "$DATA_DIR/t3_cadprompt.json" \
    --gt_dir "$DATA_DIR/gt_meshes_cadprompt" \
    --output "$RESULTS_DIR/text2cad_t3_results.json"

echo "[4/7] CADmium T1..."
python models/run_cadmium.py --test_set T1 \
    --data "$DATA_DIR/t1_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes" \
    --output "$RESULTS_DIR/cadmium_t1_results.json"

echo "[4/7] CADmium T2..."
python models/run_cadmium.py --test_set T2 \
    --data "$DATA_DIR/t2_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes_t2" \
    --output "$RESULTS_DIR/cadmium_t2_results.json"

# NOTE: CADmium T3 is N/A (prompt format mismatch)

# ── Group C: Code-based SFT/RL ─────────────────────────────────────────────────

echo "[5/7] Text-to-CadQuery T1..."
python models/run_t2cq.py --test_set T1 \
    --data "$DATA_DIR/t1_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes" \
    --output "$RESULTS_DIR/t2cq_t1_results.json"

echo "[5/7] Text-to-CadQuery T2..."
python models/run_t2cq.py --test_set T2 \
    --data "$DATA_DIR/t2_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes_t2" \
    --output "$RESULTS_DIR/t2cq_t2_results.json"

echo "[5/7] Text-to-CadQuery T3..."
python models/run_t2cq.py --test_set T3 \
    --data "$DATA_DIR/t3_cadprompt.json" \
    --gt_dir "$DATA_DIR/gt_meshes_cadprompt" \
    --output "$RESULTS_DIR/t2cq_t3_results.json"

echo "[6/7] cadrille T1..."
python models/run_cadrille.py --test_set T1 \
    --data "$DATA_DIR/t1_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes" \
    --output "$RESULTS_DIR/cadrille_t1_results.json"

echo "[6/7] cadrille T2..."
python models/run_cadrille.py --test_set T2 \
    --data "$DATA_DIR/t2_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes_t2" \
    --output "$RESULTS_DIR/cadrille_t2_results.json"

echo "[6/7] cadrille T3..."
python models/run_cadrille.py --test_set T3 \
    --data "$DATA_DIR/t3_cadprompt.json" \
    --gt_dir "$DATA_DIR/gt_meshes_cadprompt" \
    --output "$RESULTS_DIR/cadrille_t3_results.json"

echo "[7/7] ProCAD T1..."
python models/run_procad.py --test_set T1 \
    --data "$DATA_DIR/t1_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes" \
    --output "$RESULTS_DIR/procad_t1_results.json"

echo "[7/7] ProCAD T2..."
python models/run_procad.py --test_set T2 \
    --data "$DATA_DIR/t2_samples.json" \
    --gt_dir "$DATA_DIR/gt_meshes_t2" \
    --output "$RESULTS_DIR/procad_t2_results.json"

echo "[7/7] ProCAD T3..."
python models/run_procad.py --test_set T3 \
    --data "$DATA_DIR/t3_cadprompt.json" \
    --gt_dir "$DATA_DIR/gt_meshes_cadprompt" \
    --output "$RESULTS_DIR/procad_t3_results.json"

# ── Final evaluation ───────────────────────────────────────────────────────────

echo ""
echo "=== Final Evaluation ==="
python eval/evaluate_all.py --results_dir "$RESULTS_DIR" --gt_dir "$DATA_DIR"

echo ""
echo "=== Done ==="

"""
CADmium-7B Text-to-CAD inference for TextCAD-Bench.
Outputs JSON minimal representation, converted to mesh via cadmium_converter.py.

NOTE: CADmium T3 is N/A — the model outputs Python code when given
CadQuery-style prompts. Only T1 and T2 are supported.

Usage:
    python models/run_cadmium.py \
        --test_set T1 \
        --data data/t1_samples.json \
        --gt_dir data/gt_meshes \
        --output results/cadmium_t1_results.json
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))

from cadmium_converter import json_to_mesh
from metrics import evaluate_pair, with_timeout

FAILED_UIDS = {
    '0056/00564141', '0093/00939079', '0072/00721093', '0031/00316178',
    '0064/00644281', '0031/00319711', '0071/00711660', '0042/00421905',
    '0028/00281866', '0064/00649509',
}

BASE_MODEL_ID   = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_ID      = "chandar-lab/CADmium-7B"

SYSTEM_PROMPT = """Generate CAD model JSON EXACTLY matching this schema:
{"parts": {"part_1": {"coordinate_system": {"Euler Angles": [0.0,0.0,0.0],
"Translation Vector": [0.0,0.0,0.0]},
"sketch": {"face_1": {"loop_1": {"line_1": {"Start Point": [0.0,0.0],
"End Point": [0.0,0.0]}}}},
"extrusion": {"extrude_depth_towards_normal": 0.0,
"extrude_depth_opposite_normal": 0.0,
"sketch_scale": 0.0,
"operation": "NewBodyFeatureOperation"}}}}
OUTPUT ONLY RAW JSON. No explanation. No markdown."""


def generate(tokenizer, model, prompt, max_new_tokens=1024):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt}
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True)
    raw = re.sub(r'```json\n?|```\n?', '', raw).strip()
    s   = raw.find('{')
    e   = raw.rfind('}') + 1
    return raw[s:e] if s >= 0 else raw


def uid_safe(uid):
    return uid.replace('/', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', required=True, choices=['T1', 'T2'])
    parser.add_argument('--data',     required=True)
    parser.add_argument('--gt_dir',   required=True)
    parser.add_argument('--output',   required=True)
    args = parser.parse_args()

    if args.test_set == 'T3':
        print("ERROR: CADmium T3 is N/A (prompt format mismatch). "
              "Model outputs Python when given CadQuery instructions.")
        sys.exit(1)

    samples = json.load(open(args.data))

    results, done_uids = [], set()
    if os.path.exists(args.output):
        results   = json.load(open(args.output))
        done_uids = {r['uid'] for r in results}
        print(f"Resuming from {len(done_uids)} done")

    print("Loading CADmium-7B...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
    base      = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model     = PeftModel.from_pretrained(base, ADAPTER_ID)
    model     = model.merge_and_unload()
    model.eval()
    print("Model loaded.")

    for sample in tqdm(samples):
        uid = sample['uid']
        if uid in done_uids or uid in FAILED_UIDS:
            continue

        stem    = uid.split('/')[-1] if args.test_set == 'T1' else uid_safe(uid)
        gt_path = os.path.join(args.gt_dir, f"{stem}.obj")
        if not os.path.exists(gt_path):
            continue

        prompt = (sample.get('prompt_l1', sample.get('prompt', ''))
                  if args.test_set == 'T1'
                  else sample.get('prompt', '')[:400])

        raw_json  = generate(tokenizer, model, prompt)
        pred_path = None
        try:
            data = json.loads(raw_json)
            mesh = with_timeout(lambda: json_to_mesh(data, uid_safe(uid)), seconds=15)
            if mesh and not mesh.is_empty:
                pred_path = f"/tmp/cadmium_{uid_safe(uid)}.obj"
                mesh.export(pred_path)
        except Exception:
            pass

        metrics = evaluate_pair(pred_path, gt_path)
        if pred_path and os.path.exists(pred_path):
            os.remove(pred_path)

        r = {
            'uid':            uid,
            'model':          'cadmium',
            'test_set':       args.test_set,
            'complexity':     sample.get('complexity', 'unknown'),
            'n_primitives':   sample.get('n_primitives'),
            'generated_code': raw_json[:300],
            **metrics,
            'skip':           False,
        }
        results.append(r)

        if len(results) % 100 == 0:
            with open(args.output, 'w') as f:
                json.dump(results, f)
            valid = [x for x in results if x['ir_compile'] == 0]
            cds   = [x['cd_mean'] for x in valid if x['cd_mean']]
            print(f"[{len(results)}] IR={1-len(valid)/len(results):.1%}"
                  + (f" CD={np.mean(cds):.1f}" if cds else ""))

    with open(args.output, 'w') as f:
        json.dump(results, f)

    valid = [r for r in results if r['ir_compile'] == 0]
    cds   = [r['cd_mean'] for r in valid if r['cd_mean']]
    print(f"\n✅ CADmium {args.test_set}: {len(valid)}/{len(results)} valid"
          + (f" | CD={np.mean(cds):.2f}" if cds else ""))


if __name__ == '__main__':
    main()

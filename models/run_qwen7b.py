"""
Qwen-7B zero-shot Text-to-CAD inference for TextCAD-Bench.

Usage:
    python models/run_qwen7b.py \
        --test_set T1 \
        --data data/t1_samples.json \
        --gt_dir data/gt_meshes \
        --output results/qwen7b_t1_results.json

    python models/run_qwen7b.py \
        --test_set T2 \
        --data data/t2_samples.json \
        --gt_dir data/gt_meshes_t2 \
        --output results/qwen7b_t2_results.json

    python models/run_qwen7b.py \
        --test_set T3 \
        --data data/t3_cadprompt.json \
        --gt_dir data/gt_meshes_cadprompt \
        --output results/qwen7b_t3_results.json
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))

from cadquery_executor import execute_cadquery_code
from metrics import evaluate_pair, load_and_normalize

FAILED_UIDS = {
    '0056/00564141', '0093/00939079', '0072/00721093', '0031/00316178',
    '0064/00644281', '0031/00319711', '0071/00711660', '0042/00421905',
    '0028/00281866', '0064/00649509',
}

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = """You are a CadQuery expert. Generate only executable CadQuery Python code.
Import cadquery as cq. Store the final result in a variable called 'result'.
Do not include show_object() or any display calls.
Write complete code — do not truncate.
Output only Python code, no explanation, no markdown.

EXAMPLE:
import cadquery as cq
result = cq.Workplane("XY").box(1.0, 1.0, 0.5)"""


def generate(tokenizer, model, prompt, max_new_tokens=512, device='cuda'):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Generate CadQuery code for: {prompt}"}
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True)
    return re.sub(r'```python\n?|```\n?', '', generated).strip()


def get_prompt(sample, test_set, max_chars=400):
    if test_set == 'T1':
        return sample.get('prompt_l1', sample.get('prompt', ''))
    elif test_set == 'T2':
        return sample.get('prompt', '')[:max_chars]
    elif test_set == 'T3':
        return sample.get('prompt', '')
    return sample.get('prompt', '')


def uid_to_gt(uid, test_set, gt_dir):
    if test_set == 'T1':
        stem = uid.split('/')[-1]
    elif test_set == 'T2':
        stem = uid.replace('/', '_')
    else:
        stem = uid
    return os.path.join(gt_dir, f"{stem}.obj")


def uid_safe(uid):
    return uid.replace('/', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', required=True, choices=['T1', 'T2', 'T3'])
    parser.add_argument('--data',     required=True)
    parser.add_argument('--gt_dir',   required=True)
    parser.add_argument('--output',   required=True)
    parser.add_argument('--model_id', default=MODEL_ID)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    args = parser.parse_args()

    samples = json.load(open(args.data))

    # Resume support
    results, done_uids = [], set()
    if os.path.exists(args.output):
        results   = json.load(open(args.output))
        done_uids = {r['uid'] for r in results if r.get('generated_code')}
        print(f"Resuming from {len(done_uids)} done")

    print(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto").eval()
    device = next(model.parameters()).device
    print("Model loaded.")

    for sample in tqdm(samples):
        uid = sample['uid']
        if uid in done_uids or uid in FAILED_UIDS:
            continue

        gt_path = uid_to_gt(uid, args.test_set, args.gt_dir)
        if not os.path.exists(gt_path):
            continue

        prompt    = get_prompt(sample, args.test_set)
        code      = generate(tokenizer, model, prompt, args.max_new_tokens, device)
        pred_path, _ = execute_cadquery_code(code, uid_safe(uid))

        metrics = evaluate_pair(pred_path, gt_path)
        if pred_path and os.path.exists(pred_path):
            os.remove(pred_path)

        r = {
            'uid':            uid,
            'model':          'qwen7b',
            'test_set':       args.test_set,
            'complexity':     sample.get('complexity', 'unknown'),
            'n_primitives':   sample.get('n_primitives'),
            'generated_code': code,
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
    ious  = [r['iou']     for r in valid if r['iou']]
    print(f"\n✅ Done: {len(valid)}/{len(results)} valid | "
          f"IR={1-len(valid)/len(results):.1%} | "
          + (f"CD={np.mean(cds):.2f} | " if cds else "")
          + (f"IoU={np.mean(ious):.2f}" if ious else ""))


if __name__ == '__main__':
    main()

"""
cadrille (maksimko123/cadrille-rl) inference for TextCAD-Bench.
Text-only mode — no point cloud input.

Usage:
    python models/run_cadrille.py --test_set T1 \
        --data data/t1_samples.json \
        --gt_dir data/gt_meshes \
        --output results/cadrille_t1_results.json
"""

import argparse, json, os, re, sys, numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))

from cadquery_executor import execute_cadquery_code
from metrics import evaluate_pair

FAILED_UIDS = {
    '0056/00564141','0093/00939079','0072/00721093','0031/00316178',
    '0064/00644281','0031/00319711','0071/00711660','0042/00421905',
    '0028/00281866','0064/00649509',
}
MODEL_ID = "maksimko123/cadrille-rl"

SYSTEM_PROMPT = """You are a CadQuery expert that generates Python code to create 3D CAD models.
Generate only executable CadQuery code. Store the result in variable 'result'.
Output only Python code, no explanation."""

def generate(tokenizer, model, prompt, max_new_tokens=512):
    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":prompt}]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return re.sub(r'```python\n?|```\n?','',generated).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', required=True, choices=['T1','T2','T3'])
    parser.add_argument('--data',     required=True)
    parser.add_argument('--gt_dir',   required=True)
    parser.add_argument('--output',   required=True)
    args = parser.parse_args()

    samples = json.load(open(args.data))
    results, done_uids = [], set()
    if os.path.exists(args.output):
        results   = json.load(open(args.output))
        done_uids = {r['uid'] for r in results if r.get('generated_code')}
        print(f"Resuming from {len(done_uids)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto").eval()

    for sample in tqdm(samples):
        uid = sample['uid']
        if uid in done_uids or uid in FAILED_UIDS: continue
        stem = uid.split('/')[-1] if args.test_set=='T1' else uid.replace('/','_')
        gt_path = os.path.join(args.gt_dir, f"{stem}.obj")
        if not os.path.exists(gt_path): continue
        prompt = (sample.get('prompt_l1', sample.get('prompt','')) if args.test_set=='T1'
                  else sample.get('prompt','')[:400] if args.test_set=='T2'
                  else sample.get('prompt',''))
        code      = generate(tokenizer, model, prompt)
        uid_s     = uid.replace('/','_')
        pred_path,_ = execute_cadquery_code(code, uid_s)
        metrics   = evaluate_pair(pred_path, gt_path)
        if pred_path and os.path.exists(pred_path): os.remove(pred_path)
        results.append({'uid':uid,'model':'cadrille','test_set':args.test_set,
                        'complexity':sample.get('complexity','unknown'),
                        'n_primitives':sample.get('n_primitives'),
                        'generated_code':code,**metrics,'skip':False})
        if len(results)%100==0:
            with open(args.output,'w') as f: json.dump(results,f)
            valid=[x for x in results if x['ir_compile']==0]
            cds=[x['cd_mean'] for x in valid if x['cd_mean']]
            print(f"[{len(results)}] IR={1-len(valid)/len(results):.1%}"+(f" CD={np.mean(cds):.1f}" if cds else ""))

    with open(args.output,'w') as f: json.dump(results,f)
    valid=[r for r in results if r['ir_compile']==0]
    cds=[r['cd_mean'] for r in valid if r['cd_mean']]
    print(f"\n✅ cadrille {args.test_set}: {len(valid)}/{len(results)}"+(f" CD={np.mean(cds):.2f}" if cds else ""))

if __name__=='__main__': main()

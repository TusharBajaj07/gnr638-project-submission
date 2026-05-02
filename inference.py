"""GNR 638 Project 1 — inference.

Pipeline:
1. Stitch patches/* into a single map using Jigsaw V7-BB.
2. Run Qwen2.5-VL-7B-AWQ with max_pixels=2048*28*28 (validated best on dev,
   score +59.50 on 144-MCQ dev set).
3. Plain-MCQ constrained-decoding head; abstain (output 5) when top prob < 0.40.

Usage:
    python inference.py --test_dir <abs_path_to_test_dir>

Outputs `submission.csv` in the current working directory with columns
id, question_num, option (matching the sample submission format).
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---- Stitching ----
from src.stitching.jigsaw_v7_bb import JigsawV7BB


def stitch_patches(test_dir: Path) -> Image.Image:
    patches_dir = Path(test_dir) / "patches"
    paths = sorted(patches_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No patches found in {patches_dir}")
    patches_dict = {}
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            patches_dict[p.name] = img
    n = len(patches_dict)
    # Square grid heuristic (15x15 = 225 covers the sample case; adapt for others).
    grid = int(np.round(np.sqrt(n)))
    if grid * grid != n:
        # Fall back: try ceil and floor
        for cand in (grid, grid + 1, grid - 1, grid + 2, grid - 2):
            if cand > 0 and cand * cand >= n:
                grid = cand
                break
    print(f"[stitch] {n} patches, grid={grid}x{grid}", flush=True)

    if "patch_0.png" not in patches_dict:
        # Fallback anchor — pick deterministic first
        anchor = sorted(patches_dict.keys())[0]
    else:
        anchor = "patch_0.png"

    stitcher = JigsawV7BB(grid_size=(grid, grid), verbose=True, refine_iters=3)
    result = stitcher.stitch(patches_dict, anchor=anchor)
    canvas = result["image"]
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(canvas_rgb)
    print(f"[stitch] canvas={pil.size}, coverage={result.get('coverage', 0)*100:.0f}%, "
          f"time={result.get('time_sec', 0):.0f}s", flush=True)
    return pil


# ---- VLM ----

PROMPT = (
    "This is an OpenStreetMap image. Answer the multiple-choice question by "
    "responding with ONLY the number (1, 2, 3, or 4) of the correct option. "
    "If unsure, respond 5 (skip).\n\n"
    "Question: {q}\n"
    "Option 1: {o1}\n"
    "Option 2: {o2}\n"
    "Option 3: {o3}\n"
    "Option 4: {o4}"
)
PREFILL = "The correct option is: "
SKIP_THRESHOLD = 0.40


def load_vlm():
    print("[vlm] loading Qwen2.5-VL-7B-AWQ...", flush=True)
    t0 = time.time()
    import awq.modules.linear.gemm as _awq_gemm
    if hasattr(_awq_gemm, "FP16_MATMUL_HEURISTIC_CONDITION"):
        _awq_gemm.FP16_MATMUL_HEURISTIC_CONDITION = False
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor

    vlm_path = snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    cfg = AutoConfig.from_pretrained(vlm_path)
    qc = getattr(cfg, "quantization_config", None) or {}
    if isinstance(qc, dict):
        sk = list(qc.get("modules_to_not_convert") or [])
        if "lm_head" not in sk:
            sk.append("lm_head")
        qc["modules_to_not_convert"] = sk
        cfg.quantization_config = qc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path, config=cfg, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None

    # v19 winning resolution: max_pixels=2048*28*28 lifted score from
    # +52.25 (1280) to +58.50 (2048). Higher (2560/3200) regresses.
    min_px = 512 * 28 * 28
    max_px = 2048 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        vlm_path, min_pixels=min_px, max_pixels=max_px)
    print(f"[vlm] loaded in {time.time()-t0:.0f}s, "
          f"VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    return model, processor


def predict_one(model, processor, img, mcq, bare_ids_t):
    prompt = PROMPT.format(
        q=mcq.get('question', ''),
        o1=mcq.get('option_1', ''),
        o2=mcq.get('option_2', ''),
        o3=mcq.get('option_3', ''),
        o4=mcq.get('option_4', ''),
    )
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img},
                    {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    text = text + PREFILL
    inputs = processor(text=[text], images=[img], padding=True,
                       return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    opt_logits = logits[bare_ids_t]
    probs = torch.softmax(opt_logits, dim=-1)
    top_idx = int(torch.argmax(probs).item())
    top_prob = float(probs[top_idx].item())
    pred = top_idx + 1  # bare_ids ordered for digits "1".."5"
    if pred != 5 and top_prob < SKIP_THRESHOLD:
        pred = 5
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', required=True,
                    help='Directory containing patches/, test.csv, sample_submission.csv')
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    test_csv = test_dir / 'test.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found at {test_csv}")
    with open(test_csv, encoding='utf-8') as f:
        mcqs = list(csv.DictReader(f))
    print(f"[load] {len(mcqs)} MCQs from {test_csv}", flush=True)

    # Stitch patches
    stitched = stitch_patches(test_dir)

    # Load VLM
    model, processor = load_vlm()
    tok = processor.tokenizer
    bare_ids = []
    for d in ["1", "2", "3", "4", "5"]:
        b = tok.encode(d, add_special_tokens=False)
        assert len(b) == 1
        bare_ids.append(b[0])
    bare_ids_t = torch.tensor(bare_ids, device=model.device)
    print(f"[vlm] bare digit ids 1-5: {bare_ids}", flush=True)

    # Predict
    rows = []
    t_start = time.time()
    for i, mcq in enumerate(mcqs):
        try:
            pred = predict_one(model, processor, stitched, mcq, bare_ids_t)
        except Exception as e:
            print(f"[err] q{i}: {type(e).__name__}: {e}", flush=True)
            pred = 5
        # Validate: must be 1-5
        if pred not in (1, 2, 3, 4, 5):
            pred = 5
        qid = mcq.get('id', f'ques_{i+1}')
        rows.append({'id': qid, 'question_num': qid, 'option': int(pred)})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(mcqs)}] elapsed {time.time()-t_start:.0f}s", flush=True)

    # Write submission.csv (in cwd as per spec)
    out_path = Path('submission.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'question_num', 'option'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {out_path.resolve()} ({len(rows)} rows)", flush=True)


if __name__ == '__main__':
    main()

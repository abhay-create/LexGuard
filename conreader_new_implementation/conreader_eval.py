#!/usr/bin/env python3
"""
Evaluation script that writes per-example predictions (JSONL) + computes AUPR and P@0.8R.

Usage:
  python conreader_eval_with_preds.py \
    --data_dir ./data_cuad_converted \
    --model_path ./exp/best_model.pt \
    --device cpu \
    --batch_size 8 \
    --max_len 512 \
    --pred_out ./predictions.jsonl
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

# try import dataset & model from training script
try:
    from conreader_end_to_end import ContractDataset, collate_fn, ConReaderModel
except Exception as e:
    raise SystemExit("Please run this script in the same folder as conreader_end_to_end.py (so it can import ContractDataset, collate_fn, ConReaderModel). Error: " + str(e))

def compute_aupr(y_true, y_score):
    if sum(y_true) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def precision_at_recall(y_true, y_score, target_recall=0.8):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    precisions_at = [p for p,r in zip(precision, recall) if r >= target_recall]
    return max(precisions_at) if len(precisions_at) > 0 else 0.0

def _map_token_span_to_char(tokenizer: RobertaTokenizerFast, query: str, segment: str, tok_start: int, tok_end: int, max_len: int):
    """
    Given token indices (tok_start, tok_end) in the concatenated encoding (query+segment),
    compute char offsets in the segment text and return (char_start, char_end, predicted_text).
    If token indices don't exactly align to second-sequence tokens, search nearest second-sequence tokens.
    """
    enc = tokenizer(query, segment, truncation='only_second', max_length=max_len, return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    seq_ids = enc.sequence_ids()

    # Ensure indices within range
    L = len(offsets)
    if tok_start < 0 or tok_start >= L:
        tok_start = 0
    if tok_end < 0 or tok_end >= L:
        tok_end = min(L-1, tok_end)

    # If tok_start/tok_end aren't in second sequence, move them to nearest token with seq_id==1
    def find_nearest_in_seq(idx, direction=0):
        # direction = 0 prefer exact or forward, -1 search backward, +1 search forward
        if 0 <= idx < L and seq_ids[idx] == 1:
            return idx
        # search outward for nearest seq_id==1
        for d in range(1, L):
            cand1 = idx - d
            cand2 = idx + d
            if cand1 >= 0 and seq_ids[cand1] == 1:
                return cand1
            if cand2 < L and seq_ids[cand2] == 1:
                return cand2
        # fallback: find first seq_id==1
        for i, sid in enumerate(seq_ids):
            if sid == 1: return i
        return None

    s_tok = find_nearest_in_seq(tok_start)
    e_tok = find_nearest_in_seq(tok_end)
    if s_tok is None or e_tok is None:
        return 0, 0, ""  # no mapping possible

    # Offsets are relative to the second sequence (segment) for tokens with seq_id==1
    s_ch = offsets[s_tok][0]
    e_ch = offsets[e_tok][1]
    # clamp and extract
    if s_ch is None: s_ch = 0
    if e_ch is None: e_ch = 0
    s_ch = max(0, int(s_ch))
    e_ch = max(s_ch, int(e_ch))
    pred_text = segment[s_ch:e_ch]
    return s_ch, e_ch, pred_text

def evaluate_and_write_predictions(model, data_loader, tokenizer, device, args, pred_out_path: Path):
    model.eval()
    all_scores = []
    all_labels = []
    pred_out_f = open(pred_out_path, "w", encoding="utf-8")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_starts = batch['token_starts']   # cpu tensor
            token_ends = batch['token_ends']
            examples = batch['examples']           # list[ContractSegmentExample]
            start_logits, end_logits, cls, hy, def_reps = model(input_ids, attention_mask, tokenizer=tokenizer, segment_idx=0, definitions=None, clause_type=None)
            probs_s = F.softmax(start_logits, dim=-1).cpu().numpy()
            probs_e = F.softmax(end_logits, dim=-1).cpu().numpy()
            B, L = probs_s.shape

            for i in range(B):
                # find best span (simple beam: all s,e with e>=s and width<=60)
                best_score = 0.0
                best_s = 0
                best_e = 0
                max_width = min(60, L)
                for s in range(L):
                    # small optimization: if probs_s[i,s] * max(probs_e) < best_score skip - but we keep explicit for clarity
                    for e in range(s, min(s + max_width, L)):
                        sc = probs_s[i, s] * probs_e[i, e]
                        if sc > best_score:
                            best_score = sc
                            best_s = s
                            best_e = e

                # label determination (1 if any gt span present)
                label = 1 if (token_starts[i].item() != 0 or token_ends[i].item() != 0) else 0
                all_scores.append(best_score)
                all_labels.append(label)

                # Map predicted token span to char offsets and predicted text (re-encode per-example)
                ex = examples[i]
                query = ex.query
                segment_text = ex.segment_text
                # Use helper to map (best_s, best_e) -> char offsets in segment and predicted_text
                s_ch, e_ch, pred_text = _map_token_span_to_char(tokenizer, query, segment_text, best_s, best_e, args.max_len)

                # Prepare gold spans (as char spans and texts)
                gold_list = []
                for (g_s, g_e) in ex.gt_spans:
                    # guard
                    if g_s is None or g_e is None: continue
                    if g_s < 0 or g_e > len(segment_text): continue
                    gold_list.append({"start": int(g_s), "end": int(g_e), "text": segment_text[g_s:g_e]})

                # Write JSON line
                pred_record = {
                    "doc_id": ex.doc_id,
                    "segment_id": ex.segment_id,
                    "query": ex.query,
                    "query_type": ex.query_type,
                    "pred_token_span": [int(best_s), int(best_e)],
                    "pred_char_span": [int(s_ch), int(e_ch)],
                    "pred_text": pred_text,
                    "score": float(best_score),
                    "gold_spans": gold_list,
                    "label": int(label)
                }
                pred_out_f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
    pred_out_f.close()
    aupr = compute_aupr(all_labels, all_scores)
    p08 = precision_at_recall(all_labels, all_scores, 0.8)
    return aupr, p08

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help="root data dir containing 'dev/'")
    p.add_argument('--model_path', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--pred_out', type=str, default='./predictions.jsonl', help='path to write per-example predictions (jsonl)')
    args = p.parse_args()

    dev_dir = Path(args.data_dir) / 'dev'
    if not dev_dir.exists():
        raise SystemExit(f"Dev folder not found at {dev_dir}")
    dev_files = [str(dev_dir / f) for f in os.listdir(dev_dir) if f.endswith('.json')]

    device = torch.device(args.device if (torch.cuda.is_available() and 'cuda' in args.device) else 'cpu')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # collect clause types (optional; used by ConReaderModel init)
    clause_types = set()
    for jf in dev_files:
        d = json.load(open(jf,'r',encoding='utf-8'))
        for s in d.get('segments', []):
            for c in s.get('clauses', []):
                clause_types.add(c.get('type',''))
    clause_types = sorted(list(clause_types)) if len(clause_types)>0 else ['_NONE_']

    # instantiate model and load checkpoint
    model = ConReaderModel(roberta_model_name='roberta-base', reserved_slots=16, clause_types=clause_types, device=device)
    model.to(device)
    ckpt = torch.load(args.model_path, map_location='cpu')
    try:
        model.load_state_dict(ckpt, strict=False)
    except Exception as e:
        print("Warning loading state_dict with strict=False:", e)
        try:
            model.load_state_dict(ckpt)
        except Exception as e2:
            print("Warning: strict load also failed:", e2)

    # dataset + loader
    dev_ds = ContractDataset(dev_files, tokenizer, max_len=args.max_len, max_query_len=64, task='CA')
    if len(dev_ds) == 0:
        raise SystemExit("No dev examples found. Check your dev folder and files.")
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))

    pred_out_path = Path(args.pred_out)
    pred_out_path.parent.mkdir(parents=True, exist_ok=True)

    aupr, p08 = evaluate_and_write_predictions(model, dev_loader, tokenizer, device, args, pred_out_path)
    print(f"Evaluation finished. AUPR={aupr:.6f}, P@0.8R={p08:.6f}")
    print(f"Predictions written to: {pred_out_path.resolve()}")

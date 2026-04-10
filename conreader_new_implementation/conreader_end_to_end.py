"""
Fixed ConReader end-to-end script (imports bug fixed + small robustness tweaks).

Usage example:
  python3 ConReader_end_to_end.py --data_dir ./data --output_dir ./exp --device cpu --epochs 1 --batch_size 1

Dependencies:
  pip install torch transformers scikit-learn tqdm numpy regex
"""
import os
import re
import json
import math
import random
import argparse
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW                          # <-- use torch's AdamW (fixed)
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

# ---------------- Utilities ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------- Data structures ----------------
class ContractSegmentExample:
    def __init__(self, doc_id, segment_id, segment_text, query, query_type, gt_spans: List[Tuple[int,int]]):
        self.doc_id = doc_id
        self.segment_id = segment_id
        self.segment_text = segment_text
        self.query = query
        self.query_type = query_type
        self.gt_spans = gt_spans  # list of (char_start, char_end) relative to segment text

class ContractDataset(Dataset):
    def __init__(self, json_files: List[str], tokenizer: RobertaTokenizerFast, max_len=512, max_query_len=64, task='CA'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_query_len = max_query_len
        self.examples: List[ContractSegmentExample] = []
        self.task = task
        for jf in json_files:
            self._load_one(jf)

    def _load_one(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        doc_id = doc.get('doc_id', os.path.basename(json_path))
        segments = doc.get('segments', [])
        if self.task == 'CA':
            clause_types = set([c['type'] for s in segments for c in s.get('clauses', [])])
            if len(clause_types) == 0:
                clause_types = {"_NONE_"}
            for seg in segments:
                seg_id = seg['segment_id']
                seg_text = seg['text']
                clauses = seg.get('clauses', [])
                gt_spans = [(c['start'], c['end']) for c in clauses]
                for t in clause_types:
                    ex = ContractSegmentExample(doc_id, seg_id, seg_text, t, t, gt_spans if any(c.get('type')==t for c in clauses) else [])
                    self.examples.append(ex)
        else:  # CD
            for seg in segments:
                seg_id = seg['segment_id']
                seg_text = seg['text']
                clauses = seg.get('clauses', [])
                gt_spans = [(c['start'], c['end']) for c in clauses]
                # create examples pairing each seed clause text to each segment
                for s_other in [c for s in segments for c in s.get('clauses', [])]:
                    ex = ContractSegmentExample(doc_id, seg_id, seg_text, s_other['text'], s_other.get('type',''), gt_spans)
                    self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Helper to map char spans in segment text to token indices in concatenated input
def span_char_to_token_indices(tokenizer: RobertaTokenizerFast, query: str, segment: str, char_start: int, char_end: int, max_len=512, max_query_len=64):
    # We'll compute tokenization with return_offsets_mapping and use sequence_ids to pick segment tokens
    enc = tokenizer(query, segment, truncation='only_second', max_length=max_len, return_offsets_mapping=True)
    # enc is BatchEncoding; offsets for each token in the concatenated input
    offsets = enc['offset_mapping']
    # sequence_ids() gives None for special tokens, 0 for first sequence (query), 1 for second (segment)
    # To get sequence ids we call enc.sequence_ids(i) for first example (only_one)
    seq_ids = enc.sequence_ids()
    token_start_idx = None
    token_end_idx = None
    for i, (ofs, sid) in enumerate(zip(offsets, seq_ids)):
        if sid != 1:  # only segment tokens
            continue
        # ofs is (char_start, char_end) relative to the segment string (since only_second)
        s_ch, e_ch = ofs
        # the tokenizer returns offsets relative to the raw concatenated second sequence (segment)
        if s_ch <= char_start < e_ch:
            token_start_idx = i
        if s_ch < char_end <= e_ch:
            token_end_idx = i
    # fallbacks
    if token_start_idx is None:
        # find first segment token
        for i, sid in enumerate(seq_ids):
            if sid == 1:
                token_start_idx = i
                break
    if token_end_idx is None:
        token_end_idx = token_start_idx
    return token_start_idx, token_end_idx

def collate_fn(batch: List[ContractSegmentExample], tokenizer: RobertaTokenizerFast, max_len=512, max_query_len=64):
    queries = [b.query for b in batch]
    seg_texts = [b.segment_text for b in batch]
    enc = tokenizer(queries, seg_texts, padding=True, truncation='only_second', max_length=max_len, return_tensors='pt', return_attention_mask=True)
    token_starts = []
    token_ends = []
    for b in batch:
        if len(b.gt_spans) == 0:
            token_starts.append(0)
            token_ends.append(0)
        else:
            ch_s, ch_e = b.gt_spans[0]
            s_idx, e_idx = span_char_to_token_indices(tokenizer, b.query, b.segment_text, ch_s, ch_e, max_len=max_len, max_query_len=max_query_len)
            token_starts.append(s_idx if s_idx is not None else 0)
            token_ends.append(e_idx if e_idx is not None else 0)
    out = {
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'token_starts': torch.tensor(token_starts, dtype=torch.long),
        'token_ends': torch.tensor(token_ends, dtype=torch.long),
        'examples': batch
    }
    return out

# ---------------- Model ----------------
class ClauseMemory:
    def __init__(self, clause_types: List[str], per_partition_size: int = 10, device='cpu'):
        self.partition_size = per_partition_size
        self.device = device
        self.type2idx = {t: i for i, t in enumerate(clause_types)}
        self.L = len(clause_types)
        self.mem = [deque(maxlen=per_partition_size) for _ in range(self.L)]
        self.overall = deque(maxlen=self.L * per_partition_size)

    def enqueue(self, clause_type: str, hy: torch.Tensor):
        if clause_type not in self.type2idx:
            self.overall.append(hy.detach().cpu())
            return
        idx = self.type2idx[clause_type]
        self.mem[idx].append(hy.detach().cpu())
        self.overall.append(hy.detach().cpu())

    def retrieve_partition(self, clause_type: str):
        if clause_type not in self.type2idx:
            return list(self.overall)
        return list(self.mem[self.type2idx[clause_type]])

    def retrieve_all(self):
        return list(self.overall)

class ConReaderModel(nn.Module):
    def __init__(self, roberta_model_name='roberta-base', reserved_slots=16, clause_types: List[str]=None, device='cpu'):
        super().__init__()
        self.device = device
        self.reserved_slots = reserved_slots
        self.config = RobertaConfig.from_pretrained(roberta_model_name)
        self.roberta = RobertaModel.from_pretrained(roberta_model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.max_segments = 2048
        self.seg_pos_embed = nn.Embedding(self.max_segments, hidden_size)
        self.Wlcr = nn.Linear(hidden_size, hidden_size)
        self.Wy = nn.Linear(hidden_size * 2, hidden_size)
        fusion_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size*4)
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.start_proj = nn.Linear(hidden_size, 1)
        self.end_proj = nn.Linear(hidden_size, 1)
        self.margin = 1.0
        self.clause_types = clause_types if clause_types is not None else []
        self.clause_memory = ClauseMemory(self.clause_types, per_partition_size=10, device=device)
        # small projection for hy if needed
        self.hy_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward_lcr(self, input_ids, attention_mask, segment_index=0):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        # cls is at position 0
        cls = last_hidden[:, 0, :] + self.seg_pos_embed.weight[segment_index % self.max_segments].unsqueeze(0)
        return last_hidden, cls

    def encode_definitions(self, defs: List[Tuple[str,str]], tokenizer: RobertaTokenizerFast):
        reps = []
        if defs is None or len(defs) == 0:
            return torch.zeros((0, self.config.hidden_size), device=self.device)
        for k, v in defs:
            enc = tokenizer(k, v, return_tensors='pt', truncation=True, max_length=256).to(self.device)
            out = self.roberta(**enc).last_hidden_state[:, 0, :]
            reps.append(out.squeeze(0))
        return torch.stack(reps, dim=0)

    def retrieve_similar_clause(self, hlcr_cls: torch.Tensor, clause_type: Optional[str]=None):
        if clause_type is not None:
            mem = self.clause_memory.retrieve_partition(clause_type)
        else:
            mem = self.clause_memory.retrieve_all()
        if len(mem) == 0:
            return torch.zeros(self.config.hidden_size * 2, device=self.device)
        mem_tensors = torch.stack([m.to(self.device) for m in mem], dim=0)  # (N, 2H)
        proj_q = self.Wlcr(hlcr_cls.unsqueeze(0))  # (1,H)
        proj_mem = self.Wy(mem_tensors)  # (N,H)
        scores = F.cosine_similarity(proj_q, proj_mem, dim=-1)
        best_idx = torch.argmax(scores)
        return mem_tensors[best_idx].to(self.device)

    def forward(self, input_ids, attention_mask, tokenizer: RobertaTokenizerFast, segment_idx=0, definitions: List[Tuple[str,str]] = None, clause_type: Optional[str] = None):
        last_hidden, cls = self.forward_lcr(input_ids, attention_mask, segment_index=segment_idx)
        # cls: (B, H)
        B, L, H = last_hidden.size()  # last_hidden: (B, L, H)

        # Encode definitions (if any)
        if definitions is None or len(definitions) == 0:
            def_reps = torch.zeros((0, self.config.hidden_size), device=self.device)
        else:
            def_reps = self.encode_definitions(definitions, tokenizer)  # may be (N_defs, H)

        # Retrieve hy (may be vector of size 2H or zeros)
        hy = self.retrieve_similar_clause(cls, clause_type=clause_type)  # keep interface: handle hy below

        # Build relation_list as list of tensors with shape (B, H)
        relation_list = []

        # 1) cls per example (already shape (B, H))
        if cls.dim() == 1:
            cls_b = cls.unsqueeze(0).expand(B, -1)  # unlikely, but safe
        else:
            cls_b = cls  # (B, H)
        relation_list.append(cls_b)

        # 2) averaged definition reps -> (H,) -> expand to (B, H)
        if def_reps is not None and def_reps.numel() > 0:
        # def_reps: (N_defs, H) -> average to (H,)
            avg_def = def_reps.mean(dim=0)  # (H,)
            avg_def_b = avg_def.unsqueeze(0).expand(B, -1)  # (B, H)
            relation_list.append(avg_def_b)

        # 3) hy handling: hy might be 1D (2H) or already batched (rare). Ensure we project and make (B, H)
        if hy is not None and isinstance(hy, torch.Tensor) and hy.numel() > 0:
        # If hy is 1D (2H), expand to (B, 2H) then project to (B, H)
            if hy.dim() == 1:
                hy_exp = hy.unsqueeze(0).expand(B, -1).to(self.device)  # (B, 2H)
            elif hy.dim() == 2 and hy.size(0) == B:
                hy_exp = hy.to(self.device)  # already (B, 2H)
            else:
                # fallback: flatten and expand
                hy_exp = hy.view(-1).unsqueeze(0).expand(B, -1).to(self.device)
            hy_proj_b = self.hy_proj(hy_exp)  # (B, H)
            relation_list.append(hy_proj_b)

        # Build rel_tensor with shape (reserved_slots, B, H) and fill first R slots
        R = self.reserved_slots
        rel_tensor = torch.zeros((R, B, H), device=self.device)
        for i, r in enumerate(relation_list[:R]):
        # r must be (B, H) - ensure that
            if r.dim() == 1:
                r = r.unsqueeze(0).expand(B, -1)
            elif r.dim() == 2 and r.size(0) != B:
                r = r.unsqueeze(0).expand(B, -1)
            rel_tensor[i] = r.to(self.device)

        # Prepare sequence for transformer: last_hidden.transpose(0,1) -> (L, B, H)
        token_seq = last_hidden.transpose(0, 1)  # (L, B, H)
        # Concatenate relation tokens along sequence dimension -> (L+R, B, H)
        seq = torch.cat([token_seq, rel_tensor], dim=0)

        # Pass through fusion transformer (expects (seq_len, batch, H))
        fused = self.fusion(seq)  # (L+R, B, H)

        # Back to (B, L+R, H)
        fused = fused.transpose(0, 1)  # (B, L+R, H)

        # Use only token positions for start/end prediction
        token_fused = fused[:, :L, :]  # (B, L, H)

        start_logits = self.start_proj(token_fused).squeeze(-1)  # (B, L)
        end_logits = self.end_proj(token_fused).squeeze(-1)      # (B, L)

        return start_logits, end_logits, cls, hy, def_reps


# ---------------- Simple def extractor ----------------
def extract_definitions_from_text(text: str) -> List[Tuple[str,str]]:
    """
    Extract simple definitions like:
      "Term" means ... 
      Term means ...
    Returns list of (term, value).
    """

    # Pattern 1: quoted term:  "Term" means ...
    pat1 = re.compile(
        r'"([A-Za-z0-9 _\-]{1,80})"\s+'         # quoted term (space and hyphen allowed)
        r'(?:shall mean|means|shall be defined as|means the following)\s+' 
        r'([^.;\n]{10,400})[.;\n]',             # up to punctuation/newline
        flags=re.IGNORECASE
    )

    # Pattern 2: unquoted (capitalized) term: Term means ...
    # Note: hyphen put as '\-' inside class to be explicit; this avoids range parsing issues.
    pat2 = re.compile(
        r'\b([A-Z][A-Za-z0-9_\- ]{1,60})\b\s+'  # capitalized term, allow letters, digits, underscore, hyphen, space
        r'(?:means|shall mean|means the following)\s+' 
        r'([^.;\n]{10,400})[.;\n]',
        flags=re.IGNORECASE
    )

    defs = []
    for m in pat1.finditer(text):
        term = m.group(1).strip()
        val = m.group(2).strip()
        defs.append((term, val))

    for m in pat2.finditer(text):
        term = m.group(1).strip()
        val = m.group(2).strip()
        defs.append((term, val))

    # Deduplicate (case-insensitive)
    seen = set()
    out = []
    for k, v in defs:
        key = k.lower()
        if key not in seen:
            seen.add(key)
            out.append((k, v))
    return out


# ---------------- Metrics ----------------
def compute_aupr(y_true: List[int], y_score: List[float]):
    if sum(y_true) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

# ---------------- Training / Eval ----------------
def train(args):
    set_seed(args.seed)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_model)
    # prepare files
    train_path = os.path.join(args.data_dir, 'train')
    dev_path = os.path.join(args.data_dir, 'dev')
    train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.json')] if os.path.exists(train_path) else []
    dev_files   = [os.path.join(dev_path, f) for f in os.listdir(dev_path) if f.endswith('.json')] if os.path.exists(dev_path) else []

    train_ds = ContractDataset(train_files, tokenizer, max_len=args.max_len, max_query_len=args.max_query_len, task=args.task)
    dev_ds = ContractDataset(dev_files, tokenizer, max_len=args.max_len, max_query_len=args.max_query_len, task=args.task)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len, args.max_query_len))
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len, args.max_query_len))

    clause_types = set()
    for jf in train_files:
        d = json.load(open(jf,'r',encoding='utf-8'))
        for s in d.get('segments',[]):
            for c in s.get('clauses',[]):
                clause_types.add(c.get('type',''))
    clause_types = sorted(list(clause_types)) if len(clause_types)>0 else ['_NONE_']

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    model = ConReaderModel(roberta_model_name=args.roberta_model, reserved_slots=args.reserved_slots, clause_types=clause_types, device=device)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    best_dev = -1.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_starts = batch['token_starts'].to(device)
            token_ends = batch['token_ends'].to(device)
            examples = batch['examples']
            optimizer.zero_grad()
            # definitions: try to load from same doc file if present
            defs = []
            if len(examples) > 0:
                ex = examples[0]
                docfile = os.path.join(args.data_dir, 'train', f"{ex.doc_id}.json")
                if os.path.exists(docfile):
                    doc = json.load(open(docfile,'r',encoding='utf-8'))
                    defs = extract_definitions_from_text(doc.get('text',''))
            start_logits, end_logits, cls, hy, def_reps = model(input_ids, attention_mask, tokenizer=tokenizer, segment_idx=0, definitions=defs, clause_type=None)
            B, L = start_logits.size()
            gold_s = token_starts.clamp(0, L-1)
            gold_e = token_ends.clamp(0, L-1)
            loss_s = F.cross_entropy(start_logits, gold_s)
            loss_e = F.cross_entropy(end_logits, gold_e)
            Le = (loss_s + loss_e) / 2.0
            loss = Le
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        dev_score = evaluate(model, dev_loader, tokenizer, device, args)
        print(f"Epoch {epoch} dev_score AUPR={dev_score:.4f}")
        if dev_score > best_dev:
            best_dev = dev_score
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
    print("Training finished. Best dev:", best_dev)

def evaluate(model: ConReaderModel, data_loader: DataLoader, tokenizer: RobertaTokenizerFast, device: torch.device, args):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_starts = batch['token_starts']
            token_ends = batch['token_ends']
            examples = batch['examples']
            start_logits, end_logits, cls, hy, def_reps = model(input_ids, attention_mask, tokenizer=tokenizer, segment_idx=0, definitions=None, clause_type=None)
            B,L = start_logits.size()
            probs_s = F.softmax(start_logits, dim=-1).cpu().numpy()
            probs_e = F.softmax(end_logits, dim=-1).cpu().numpy()
            for i in range(B):
                best_score = 0.0
                # small beam to avoid O(L^2) - limit span width
                for s in range(L):
                    for e in range(s, min(s+60, L)):
                        sc = probs_s[i,s] * probs_e[i,e]
                        if sc > best_score:
                            best_score = sc
                label = 1 if (token_starts[i].item() != 0 or token_ends[i].item() != 0) else 0
                all_scores.append(best_score)
                all_labels.append(label)
    return compute_aupr(all_labels, all_scores)

# ---------------- CLI ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--roberta_model', type=str, default='roberta-base')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--max_query_len', type=int, default=64)
    parser.add_argument('--reserved_slots', type=int, default=16)
    parser.add_argument('--task', type=str, default='CA', choices=['CA','CD'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

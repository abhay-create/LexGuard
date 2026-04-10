"""
convert_three_cuad_to_conreader.py

Reads the following files (if present) from current directory:
 - train_separate_questions.json  -> treated as training split
 - test.json                      -> treated as dev split
 - CUADv1.json                    -> optional additional split (will be merged into train if not otherwise used)

Outputs:
 - <outdir>/train/<doc_id>.json
 - <outdir>/dev/<doc_id>.json

Each output JSON follows ConReader expected schema:
{
  "doc_id": "<id>",
  "text": "<full contract text>",
  "segments": [
    {"segment_id": 0, "text": "<paragraph text>", "clauses": [ {"start":<int>,"end":<int>,"text":"...","type":"<question>"} ]}
  ]
}
"""
import json, os, argparse
from pathlib import Path
from typing import List, Dict, Any

def parse_squad_style_dataset(entries) -> Dict[str, Dict]:
    """
    Parse a SQuAD-like list of article entries.
    Returns mapping doc_id -> {"full_text":..., "segments":[...]}
    """
    docs = {}
    for art in entries:
        # art may be an article with 'title' and 'paragraphs' or a paragraph-level dict with 'context' + 'qas'
        if isinstance(art, dict) and 'title' in art and 'paragraphs' in art:
            title = art.get('title') or art.get('doc_id') or "unknown"
            paragraphs = art.get('paragraphs', [])
            full_parts = []
            segments = []
            for i, p in enumerate(paragraphs):
                context = p.get('context', '') or ""
                full_parts.append(context)
                clauses = []
                for qa in p.get('qas', []):
                    qtext = qa.get('question', qa.get('id', 'unknown'))
                    for ans in qa.get('answers', []):
                        a_text = ans.get('text', '')
                        a_start = ans.get('answer_start', ans.get('start', 0))
                        try:
                            a_start = int(a_start)
                        except Exception:
                            a_start = 0
                        a_end = a_start + len(a_text)
                        # clamp
                        if a_start < 0: a_start = 0
                        if a_end > len(context): a_end = min(a_end, len(context))
                        clauses.append({"start": a_start, "end": a_end, "text": a_text, "type": qtext})
                segments.append({"segment_id": i, "text": context, "clauses": clauses})
            docs[title] = {"full_text": "\n\n".join(full_parts), "segments": segments}
        elif isinstance(art, dict) and 'context' in art:
            # paragraph-level row: try to get title/doc_id from art if present
            title = art.get('title') or art.get('doc_id') or art.get('article_title') or "unknown"
            context = art.get('context', '') or ""
            qas = art.get('qas', art.get('qas', [])) or []
            clauses = []
            for qa in qas:
                qtext = qa.get('question', qa.get('id', 'unknown'))
                for ans in qa.get('answers', []):
                    a_text = ans.get('text', '')
                    a_start = ans.get('answer_start', ans.get('start', 0))
                    try:
                        a_start = int(a_start)
                    except Exception:
                        a_start = 0
                    a_end = a_start + len(a_text)
                    if a_start < 0: a_start = 0
                    if a_end > len(context): a_end = min(a_end, len(context))
                    clauses.append({"start": a_start, "end": a_end, "text": a_text, "type": qtext})
            # if title exists already, append segment to existing doc, else create new
            if title in docs:
                sid = len(docs[title]["segments"])
                docs[title]["segments"].append({"segment_id": sid, "text": context, "clauses": clauses})
                docs[title]["full_text"] += "\n\n" + context
            else:
                docs[title] = {"full_text": context, "segments": [{"segment_id": 0, "text": context, "clauses": clauses}]}
        elif isinstance(art, dict) and 'data' in art and isinstance(art['data'], list):
            # top-level wrapper
            sub = parse_squad_style_dataset(art['data'])
            for k,v in sub.items():
                if k in docs:
                    # merge segments
                    base = docs[k]
                    offset = len(base['segments'])
                    for seg in v['segments']:
                        seg['segment_id'] = offset + seg['segment_id']
                        base['segments'].append(seg)
                    base['full_text'] += "\n\n" + v['full_text']
                else:
                    docs[k] = v
        else:
            # unknown record shape; skip safely
            continue
    return docs

def load_json_if_exists(p: Path):
    if not p.exists():
        return []
    try:
        j = json.load(open(p, 'r', encoding='utf-8'))
    except Exception as e:
        print("Failed to load",p,"->",e)
        return []
    # normalize to list of article-like dicts
    if isinstance(j, dict) and 'data' in j and isinstance(j['data'], list):
        return j['data']
    if isinstance(j, list):
        return j
    if isinstance(j, dict) and 'paragraphs' in j:
        return [j]
    # fallback: wrap single dict
    return [j]

def save_contract_json(outdir: Path, doc_id: str, full_text: str, segments: List[Dict[str,Any]]):
    safe = str(doc_id).replace("/", "_").replace(" ", "_")[:200]
    out = {"doc_id": safe, "text": full_text, "segments": segments}
    fname = outdir / f"{safe}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main(args):
    cwd = Path(".")
    train_file = cwd / "train_separate_questions.json"
    dev_file = cwd / "test.json"
    extra_file = cwd / "CUADv1.json"

    train_entries = load_json_if_exists(train_file)
    dev_entries = load_json_if_exists(dev_file)
    extra_entries = load_json_if_exists(extra_file)

    print("Loaded counts: train entries:", len(train_entries), "dev entries:", len(dev_entries), "extra entries:", len(extra_entries))

    # If train empty but extra present, use extra as train
    if len(train_entries) == 0 and len(extra_entries) > 0:
        train_entries = extra_entries
        extra_entries = []

    # Prepare output dirs
    outdir = Path(args.outdir)
    train_out = outdir / "train"; dev_out = outdir / "dev"
    train_out.mkdir(parents=True, exist_ok=True)
    dev_out.mkdir(parents=True, exist_ok=True)

    # Parse and write
    train_docs = parse_squad_style_dataset(train_entries)
    dev_docs = parse_squad_style_dataset(dev_entries)

    for doc_id, doc in train_docs.items():
        save_contract_json(train_out, doc_id, doc["full_text"], doc["segments"])
    for doc_id, doc in dev_docs.items():
        save_contract_json(dev_out, doc_id, doc["full_text"], doc["segments"])

    print("Wrote train files:", len(list(train_out.glob("*.json"))))
    print("Wrote dev files:  ", len(list(dev_out.glob("*.json"))))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./data_cuad_converted", help="output directory")
    args = parser.parse_args()
    main(args)
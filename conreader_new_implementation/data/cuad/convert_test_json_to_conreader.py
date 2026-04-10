#!/usr/bin/env python3
"""
convert_test_json_to_conreader.py

Convert a CUAD-format test.json (SQuAD-style) into per-document JSON files
matching the ConReader expected schema.

Input:  test.json  (SQuAD-style structure: { "data": [ { "title":..., "paragraphs":[ {"context":..., "qas":[...]} ] }, ... ] })
Output: <outdir>/dev/<DOC_ID>.json

Output JSON schema (per file):
{
  "doc_id": "<doc id>",
  "title": "<title>",
  "text": "<full concatenated contract text>",
  "segments": [
    {
      "segment_id": "<segment id>",
      "text": "<paragraph text>",
      "clauses": [
         {"start": int, "end": int, "text": "<answer text>", "type": "<question-or-id>"}
      ]
    },
    ...
  ]
}

Usage:
  python convert_test_json_to_conreader.py --input test.json --outdir ./data_cuad_converted

"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def convert_test_json(input_path: Path, outdir: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    with input_path.open("r", encoding="utf-8") as f:
        j = json.load(f)

    data = j.get("data", [])
    out_dev = outdir / "dev"
    out_dev.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    seg_count = 0
    clause_count = 0
    written_files = []

    for art in data:
        # Title (use unique, safe filename)
        title = art.get("title") or art.get("doc_id") or f"doc_{doc_count}"
        safe_title = str(title).replace("/", "_")[:150]

        # Build segments and clauses
        paragraphs = art.get("paragraphs", [])
        segments = []
        full_text_parts = []
        for p_idx, p in enumerate(paragraphs):
            context = p.get("context", "") or ""
            full_text_parts.append(context)
            seg_id = f"{safe_title}__seg{p_idx}"
            clauses = []
            for qa in p.get("qas", []):
                # determine clause type: prefer QA id suffix after '__', else question text
                qid = qa.get("id", "")
                if isinstance(qid, str) and "__" in qid:
                    clause_type = qid.split("__")[-1]
                else:
                    clause_type = qa.get("question", "").strip()[:120]
                answers = qa.get("answers", [])
                for ans in answers:
                    atext = ans.get("text", "")
                    astart = ans.get("answer_start", ans.get("start", None))
                    # validate start
                    if astart is None:
                        continue
                    try:
                        astart = int(astart)
                    except Exception:
                        continue
                    if astart < 0 or astart >= len(context):
                        # invalid offset relative to paragraph: skip
                        continue
                    aend = astart + len(atext)
                    if aend > len(context):
                        aend = len(context)
                    clauses.append({
                        "start": astart,
                        "end": aend,
                        "text": atext,
                        "type": clause_type
                    })
                    clause_count += 1
            segments.append({
                "segment_id": seg_id,
                "text": context,
                "clauses": clauses
            })
            seg_count += 1

        doc_obj = {
            "doc_id": safe_title,
            "title": title,
            "text": "\n\n".join(full_text_parts),
            "segments": segments
        }

        out_file = out_dev / f"{safe_title}.json"
        with out_file.open("w", encoding="utf-8") as wf:
            json.dump(doc_obj, wf, ensure_ascii=False, indent=2)
        written_files.append(out_file.name)
        doc_count += 1

    print(f"Converted {doc_count} documents -> wrote {len(written_files)} files to {out_dev}")
    print(f"Segments: {seg_count}, Clauses (answers) extracted: {clause_count}")
    if len(written_files) > 0:
        print("Sample files:", written_files[:8])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="test.json", help="Path to CUAD test.json")
    p.add_argument("--outdir", type=str, default="./data_cuad_converted", help="Output directory")
    args = p.parse_args()
    convert_test_json(Path(args.input), Path(args.outdir))

if __name__ == "__main__":
    main()

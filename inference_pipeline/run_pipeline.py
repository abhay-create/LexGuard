#!/usr/bin/env python3
import os
import json
import argparse
from inference_pipeline.clause_extractor import extract_clauses
from inference_pipeline.compliance_checker import compare_clauses, save_compliance_report



def main():
    parser = argparse.ArgumentParser(description="Run ConReader + Ollama offline compliance pipeline.")
    parser.add_argument("--pdf1", required=True, help="Path to first contract PDF.")
    parser.add_argument("--pdf2", required=True, help="Path to second contract PDF.")
    parser.add_argument("--model_path", default="../conreader_new_implementation/exp_test/best_model.pt",
                        help="Path to trained ConReader checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda'.")
    parser.add_argument("--llm_model", default="tinyllama", help="Ollama model (llama3, mistral, etc).")
    parser.add_argument("--out_dir", default="./outputs", help="Output directory.")
    parser.add_argument("--out_name", default="compliance_report.json", help="Output filename.")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip extraction if *_clauses.json already exist.")
    args = parser.parse_args()

    pdf1, pdf2 = os.path.abspath(args.pdf1), os.path.abspath(args.pdf2)
    model_path, out_dir = os.path.abspath(args.model_path), os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)

    base1 = os.path.splitext(os.path.basename(pdf1))[0]
    base2 = os.path.splitext(os.path.basename(pdf2))[0]
    json1 = os.path.join(out_dir, f"{base1}_clauses.json")
    json2 = os.path.join(out_dir, f"{base2}_clauses.json")

    if not args.skip_extraction:
        print(f"🔍 Extracting clauses from:\n  1️⃣ {pdf1}\n  2️⃣ {pdf2}")
        doc1 = extract_clauses(pdf1, model_path, device=args.device)
        doc2 = extract_clauses(pdf2, model_path, device=args.device)
        json.dump(doc1, open(json1, "w"), indent=2)
        json.dump(doc2, open(json2, "w"), indent=2)
        print(f"✅ Clause JSONs saved to:\n  - {json1}\n  - {json2}")
    else:
        print("⏩ Using existing clause JSONs...")
        doc1, doc2 = json.load(open(json1)), json.load(open(json2))

    print(f"⚖️ Running clause comparison via Ollama model: {args.llm_model}")
    results = compare_clauses(doc1, doc2, out_file = out_path, model_name = args.llm_model)
    print(f"✅ Done! Compliance report → {out_path}")


if __name__ == "__main__":
    main()

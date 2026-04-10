import re
import textwrap
import torch
import torch.nn.functional as F
from pypdf import PdfReader
from transformers import RobertaTokenizer
from conreader_new_implementation.conreader_end_to_end import ConReaderModel


# ------------------------------------------------------------
# STEP 1 — Extract text from PDF using PyPDF
# ------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file using PyPDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------------------------------
# STEP 2 — Split text into ~clause-sized segments
# ------------------------------------------------------------
def pdf_to_segments(pdf_path):
    """Split long text into ~512-token chunks for model input."""
    text = extract_text_from_pdf(pdf_path)
    chunks = textwrap.wrap(text, 1500)  # roughly 512 tokens
    segments = [{"segment_id": i, "text": c} for i, c in enumerate(chunks) if len(c) > 80]
    print(f"📄 Extracted {len(segments)} segments from {pdf_path}")
    return segments


# ------------------------------------------------------------
# STEP 3 — Wrapper to get predictions (no .predict() method)
# ------------------------------------------------------------
def run_prediction(model, tokenizer, text, clause_types, device="cpu"):
    """Run clause-type classification for one text segment."""
    results = {}
    for clause_type in clause_types:
        inputs = tokenizer(
            clause_type, text,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            start_logits, end_logits, cls, hy, def_reps = model(**inputs, tokenizer=tokenizer)

        # Softmax over start/end logits for span confidence
        probs_s = F.softmax(start_logits, dim=-1)
        probs_e = F.softmax(end_logits, dim=-1)
        best_s = torch.argmax(probs_s)
        best_e = torch.argmax(probs_e)
        confidence = (probs_s[0, best_s] * probs_e[0, best_e]).item()
        results[clause_type] = confidence
    return results


# ------------------------------------------------------------
# STEP 4 — Clause extraction logic
# ------------------------------------------------------------
def extract_clauses(pdf_path, model_path, device="cpu"):
    """Extract clauses from a PDF using fine-tuned ConReader."""
    print(f"🧠 Loading ConReader model from: {model_path}")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    model = ConReaderModel(device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Clause labels (CUAD categories)
    clause_types = [
        "Confidentiality", "Governing Law", "Indemnification",
        "Termination", "Liability Cap", "Force Majeure",
        "Warranty", "Payment Terms", "Assignment",
        "Compliance with Laws", "Insurance", "IP Ownership",
        "Data Protection", "Audit Rights", "Non Compete",
        "Dispute Resolution", "Change of Control", "Publicity",
        "Subcontracting", "Severability", "Entire Agreement",
        "Survival", "Notices", "Independent Contractor",
        "Exclusivity", "Anti Bribery", "Waiver"
    ]

    segments = pdf_to_segments(pdf_path)
    clause_dict = {}

    print(f"🔍 Running clause detection on {len(segments)} segments...")

    for seg in segments:
        text = seg["text"]
        try:
            preds = run_prediction(model, tokenizer, text, clause_types, device=device)
            # threshold = 0.5 → mark present
            for label, score in preds.items():
                if score > 0.5:
                    clause_dict.setdefault(label, []).append(text)
        except Exception as e:
            print(f"⚠️ Segment {seg['segment_id']} skipped: {e}")
            continue

    total = sum(len(v) for v in clause_dict.values())
    print(f"✅ Extracted {total} clauses across {len(clause_dict)} types from {pdf_path}")
    return clause_dict

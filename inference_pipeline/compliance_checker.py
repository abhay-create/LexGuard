import subprocess
from tqdm import tqdm
import json
import re
import time
import os
import tempfile

def run_ollama(prompt, model="tinyllama", timeout=30, retries=1, retry_delay=1.0):
    """Run offline Ollama LLM for clause comparison.
    Returns the model's stdout (str). Does minimal retry on failure/empty output.
    """
    cmd = ["ollama", "run", model, prompt]
    for attempt in range(retries + 1):
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            out = (res.stdout or "").strip()
            if out:
                return out
        except subprocess.TimeoutExpired:
            out = ""
        if attempt < retries:
            time.sleep(retry_delay)
    return out

def parse_result_line(line):
    """Parse the single-line response expected from the LLM.
    Expected exact formats (one line only):
      - COMPLIANT
      - NON-COMPLIANT: <brief reason>
      - UNRELATED

    Returns tuple (result_keyword, reason_or_None)
    """
    if not line:
        return "UNRELATED", None

    l = line.strip()
    upper = l.upper()

    if upper == "COMPLIANT":
        return "COMPLIANT", None
    if upper == "UNRELATED":
        return "UNRELATED", None

    m = re.match(r'^\s*NON[- ]?COMPLIANT\s*[:\-—]\s*(.+)$', l, flags=re.IGNORECASE)
    if m:
        reason = m.group(1).strip()
        reason = reason.split('\n', 1)[0].strip()
        if len(reason) > 300:
            reason = reason[:300].rsplit(' ', 1)[0] + "…"
        return "NON-COMPLIANT", reason

    if "NON" in upper and "COMPLIANT" in upper:
        parts = re.split(r'NON[- ]?COMPLIANT', l, flags=re.IGNORECASE)
        if len(parts) > 1:
            reason = parts[1].lstrip(" :\-—").strip()
            if reason:
                return "NON-COMPLIANT", reason
        return "NON-COMPLIANT", None

    return "UNRELATED", None

def _atomic_write_json(data, out_file):
    """Write JSON atomically: write to temp file then replace target."""
    dirpath = os.path.dirname(os.path.abspath(out_file)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            json.dump(data, tmpf, indent=2, ensure_ascii=False)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        os.replace(tmp_path, out_file)
    finally:
        # cleanup if something went wrong and temp still exists
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def save_compliance_report(results, out_file="compliance_report.json"):
    """Save results to JSON file atomically."""
    _atomic_write_json(results, out_file)
    # small console feedback so user sees progress
    print(f"✅ Autosaved compliance report to {out_file}")

def compare_clauses(doc1, doc2, model_name="tinyllama", out_file="compliance_report.json"):
    """Compare same-type clauses between two extracted documents and autosave after every pair.

    doc1, doc2: dict mapping clause_type -> list of clause strings
    Returns dict: clause_type -> list of comparison results where each result is:
      {
        "doc1_clause": "...",
        "doc2_clause": "...",
        "result": "COMPLIANT" | "NON-COMPLIANT" | "UNRELATED",
        # "reason": "..."  # present only for NON-COMPLIANT
      }
    """
    results = {}
    common_clause_types = set(doc1.keys()) & set(doc2.keys())

    print(f"\n⚖️ Comparing {len(common_clause_types)} common clause types between documents...\n")

    base_prompt_template = """You are a legal compliance analyst.
Compare the following two clauses of type "{clause_type}" from two contracts.

INSTRUCTIONS (critical):
- Output exactly ONE LINE and NOTHING ELSE.
- The output must be one of the following three formats (case-insensitive):
    1) COMPLIANT
    2) NON-COMPLIANT
    3) UNRELATED
- Do NOT output any additional explanation, analysis, or extra lines.

Clause from Document 1:
{c1}

Clause from Document 2:
{c2}
"""

    for clause_type in tqdm(sorted(common_clause_types), desc="Clause Types", ncols=100):
        c1s, c2s = doc1[clause_type], doc2[clause_type]
        clause_pairs = [(c1, c2) for c1 in c1s for c2 in c2s]
        clause_results = results.setdefault(clause_type, [])

        for c1, c2 in tqdm(clause_pairs, leave=False, desc=f"Comparing {clause_type}", ncols=100):
            prompt = base_prompt_template.format(clause_type=clause_type, c1=c1.strip(), c2=c2.strip())
            analysis = run_ollama(prompt, model=model_name, retries=1, retry_delay=0.5)
            first_line = (analysis.splitlines()[0].strip() if analysis else "")
            result_keyword, reason = parse_result_line(first_line)

            entry = {
                "doc1_clause": c1,
                "doc2_clause": c2,
                "result": result_keyword
            }
            # if reason:
            #     entry["reason"] = reason

            clause_results.append(entry)

            # --- Incremental save AFTER every clause comparison ---
            save_compliance_report(results, out_file=out_file)

    print("\n✅ Clause comparison complete.\n")
    return results
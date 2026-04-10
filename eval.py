import json
import sys
from difflib import SequenceMatcher

# Try to import tqdm for the progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
    def tqdm(iterable, **kwargs):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        desc = kwargs.get('desc', 'Processing')
        print(f"Starting: {desc}")
        for i, item in enumerate(iterable):
            if total and i % 10 == 0:
                sys.stdout.write(f"\r{desc}: {i}/{total}")
                sys.stdout.flush()
            yield item
        print(f"\r{desc}: Complete!        ")

def safe_print(msg):
    if HAS_TQDM:
        tqdm.write(msg)
    else:
        print(msg)

def calculate_overlap(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def evaluate_compliance(report_path, changes_path, threshold=0.75):
    """
    threshold: The minimum similarity required for BOTH original AND modified 
               clauses individually to consider it a match.
    """
    
    # 1. Load Data
    print(f"Loading report from: {report_path}")
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    print(f"Loading ground truth from: {changes_path}")
    with open(changes_path, 'r', encoding='utf-8') as f:
        changes_data = json.load(f)

    # 2. Filter Predictions
    # We only look at predictions the model flagged as "NON-COMPLIANT"
    non_compliant_predictions = []
    
    for category, items in report_data.items():
        for item in items:
            # Normalize check to handle "Non Compliant", "NON-COMPLIANT", etc.
            res = item.get('result', '').upper().replace("-", " ")
            if "NON COMPLIANT" in res:
                item['category'] = category
                non_compliant_predictions.append(item)

    print(f"Loaded {len(changes_data)} ground truth items.")
    print(f"Search Space: {len(non_compliant_predictions)} 'NON-COMPLIANT' predictions.")
    print(f"Strict Matching Threshold: {threshold*100}% per clause.\n")

    metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
    }

    # 3. Iterate through Ground Truth
    for gt_item in tqdm(changes_data, desc="Evaluating", unit="clause"):
        gt_original = gt_item.get('original', '')
        gt_modified = gt_item.get('ollama_generated', '') or gt_item.get('modified', '')
        
        # GT: kept=True -> NON-COMPLIANT
        gt_is_non_compliant = gt_item.get('kept') is True
        gt_label_str = "NON-COMPLIANT" if gt_is_non_compliant else "COMPLIANT"

        best_match = None
        best_combined_score = 0
        match_scores = (0, 0) # Store (orig_score, mod_score)

        # 4. Strict Matching Loop
        for pred_item in non_compliant_predictions:
            pred_doc1 = pred_item.get('doc1_clause', '')
            
            # CHECK 1: Original Clause
            score_1 = calculate_overlap(gt_original, pred_doc1)
            if score_1 < threshold: 
                continue # Fail fast: Original texts don't match enough

            pred_doc2 = pred_item.get('doc2_clause', '')
            
            # CHECK 2: Modified Clause
            score_2 = calculate_overlap(gt_modified, pred_doc2)
            if score_2 < threshold:
                continue # Fail fast: Modified texts don't match enough
            
            # If we are here, BOTH thresholds are met.
            # We use sum/avg only to pick the *best* candidate if multiple exist.
            combined = score_1 + score_2
            if combined > best_combined_score:
                best_combined_score = combined
                best_match = pred_item
                match_scores = (score_1, score_2)

        # 5. Evaluate Result
        match_found = best_match is not None

        if match_found:
            safe_print("-" * 60)
            safe_print(f"MATCH FOUND (Orig: {match_scores[0]:.2f}, Mod: {match_scores[1]:.2f})")
            safe_print(f"GT Orig:   {gt_original}...")
            safe_print(f"Pred Orig: {best_match.get('doc1_clause', '')}...")
            
            if gt_is_non_compliant:
                metrics["true_positives"] += 1
                safe_print("Result:    ✅ TRUE POSITIVE")
            else:
                metrics["false_positives"] += 1
                safe_print(f"Result:    ❌ FALSE POSITIVE (GT was {gt_label_str})")
        else:
            if gt_is_non_compliant:
                metrics["false_negatives"] += 1
                safe_print("\n" + "-" * 60)
                safe_print(f"MISSED DETECTION (No candidate met strict {threshold} threshold)")
                safe_print(f"Snippet:   {gt_original[:100]}...")
                safe_print("Result:    ⚠️ FALSE NEGATIVE")
            else:
                metrics["true_negatives"] += 1

    # 6. Final Calculation
    tp = metrics["true_positives"]
    fp = metrics["false_positives"]
    tn = metrics["true_negatives"]
    fn = metrics["false_negatives"]
    
    total = tp + fp + tn + fn
    
    try:
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    except ZeroDivisionError:
        accuracy = precision = recall = f1 = 0.0

    print("\n" + "="*30)
    print("STRICT EVALUATION RESULTS")
    print("="*30)
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"False Negatives: {fn}")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_compliance(
        './outputs/compliance_report.json', 
        r'./annotated/Transportation/PenntexMidstreamPartnersLp_20150416_S-1A_EX-10.4_9042833_EX-10.4_Transportation Agreement/changes.json', 
        threshold=0.07
    )
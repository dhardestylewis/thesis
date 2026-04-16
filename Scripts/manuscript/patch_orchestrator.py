import re

# We will inject compute_ace after compute_ece
ace_func = """
def compute_ace(y_true, y_prob, n_bins=10):
    sorted_idx = np.argsort(y_prob)
    y_prob_sorted = y_prob[sorted_idx]
    y_true_sorted = y_true[sorted_idx]
    bin_size = max(1, len(y_prob) // n_bins)
    ace = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_prob)
        bin_prob = y_prob_sorted[start:end]
        bin_true = y_true_sorted[start:end]
        if len(bin_prob) > 0:
            ace += (len(bin_prob) / len(y_prob)) * abs(bin_prob.mean() - bin_true.mean())
    return float(ace)
"""

scripts = [
    r'Analysis\Scripts\Modeling\Production_Models\run_multi_horizon.py',
    r'Analysis\Scripts\Modeling\Production_Models\run_alternative_architectures.py',
    r'Analysis\Scripts\Modeling\Production_Models\run_calibration_benchmark.py'
]

for fp in scripts:
    with open(fp, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Inject compute_ace
    if 'def compute_ace' not in text:
        text = text.replace('def compute_ece(y_true, y_prob, n_bins=10):', 
                            ace_func + "\ndef compute_ece(y_true, y_prob, n_bins=10):")
                            
    # Specific patching per-file
    if 'run_multi_horizon' in fp:
        if 'ACE_Post' not in text:
            text = text.replace("ece_pre = compute_ece(y_test, preds_uncalibrated)",
                                "ece_pre = compute_ece(y_test, preds_uncalibrated)\n    ace_pre = compute_ace(y_test, preds_uncalibrated)")
            text = text.replace("ece_post = compute_ece(y_test, preds_calibrated)",
                                "ece_post = compute_ece(y_test, preds_calibrated)\n    ace_post = compute_ace(y_test, preds_calibrated)")
            
            text = text.replace("'ECE_Pre': round(ece_pre, 4) if not np.isnan(ece_pre) else None,",
                                "'ECE_Pre': round(ece_pre, 4) if not np.isnan(ece_pre) else None,\n        'ACE_Pre': round(ace_pre, 4),")
            text = text.replace("'ECE_Post': round(ece_post, 4) if not np.isnan(ece_post) else None,",
                                "'ECE_Post': round(ece_post, 4) if not np.isnan(ece_post) else None,\n        'ACE_Post': round(ace_post, 4),")
                                
            text = text.replace(r"\textbf{ECE (Pre)} & \textbf{ECE (Post)} \\", 
                                r"\textbf{ECE (Pre)} & \textbf{ECE (Post)} & \textbf{ACE (Post)} \\")
            text = text.replace(r"\begin{tabular}{lccccc}", r"\begin{tabular}{lcccccc}")
            
            text = text.replace(r"ece_post_str = f\"{r['ECE_Post']:.3f}\" if r['ECE_Post'] is not None else \"---\"",
                                "ece_post_str = f\"{r['ECE_Post']:.3f}\" if r['ECE_Post'] is not None else \"---\"\n        ace_post_str = f\"{r['ACE_Post']:.3f}\" if r.get('ACE_Post') is not None else \"---\"")
                                
            text = text.replace("ece_pre_str} & {ece_post_str} \\\\", "ece_pre_str} & {ece_post_str} & {ace_post_str} \\\\")

    elif 'run_alternative_architectures' in fp:
        if 'ACE (Cal)' not in text:
            # Inject ACE calls
            text = text.replace("ece_pre = compute_ece(y_test, preds_raw)", "ece_pre = compute_ece(y_test, preds_raw)\n            ace_pre = compute_ace(y_test, preds_raw)")
            text = text.replace("ece_post = compute_ece(y_test, preds_cal)", "ece_post = compute_ece(y_test, preds_cal)\n            ace_post = compute_ace(y_test, preds_cal)")
            text = text.replace("ece_pre_sm = compute_ece(y_test, preds_smote_raw)", "ece_pre_sm = compute_ece(y_test, preds_smote_raw)\n            ace_pre_sm = compute_ace(y_test, preds_smote_raw)")
            text = text.replace("ece_post_sm = compute_ece(y_test, preds_smote_cal)", "ece_post_sm = compute_ece(y_test, preds_smote_cal)\n            ace_post_sm = compute_ace(y_test, preds_smote_cal)")
            text = text.replace("ece_pre_tab = compute_ece(y_test, preds_tabnet_raw)", "ece_pre_tab = compute_ece(y_test, preds_tabnet_raw)\n        ace_pre_tab = compute_ace(y_test, preds_tabnet_raw)")
            text = text.replace("ece_post_tab = compute_ece(y_test, preds_tabnet_cal)", "ece_post_tab = compute_ece(y_test, preds_tabnet_cal)\n        ace_post_tab = compute_ace(y_test, preds_tabnet_cal)")
            
            # Update metric dicts
            text = text.replace("'ECE (Raw)': ece_pre,", "'ECE (Raw)': ece_pre, 'ACE (Cal)': ace_post,")
            text = text.replace("'ECE (Raw)': ece_pre_sm,", "'ECE (Raw)': ece_pre_sm, 'ACE (Cal)': ace_post_sm,")
            text = text.replace("'ECE (Raw)': ece_pre_tab,", "'ECE (Raw)': ece_pre_tab, 'ACE (Cal)': ace_post_tab,")
            
            # Update tex lines
            text = text.replace(r"\textbf{ECE (Raw)} & \textbf{ECE (Cal)} \\", r"\textbf{ECE (Cal)} & \textbf{ACE (Cal)} \\")
            
            text = text.replace("tex_lines.append(f\"{horizon_clean} & {arch_clean} & {pr} & {roc} & {ece_raw} & {ece_cal} \\\\\\\\\")",
                                "ace_cal = f\"{row.get('ACE (Cal)', 0):.3f}\"\n        tex_lines.append(f\"{horizon_clean} & {arch_clean} & {pr} & {roc} & {ece_cal} & {ace_cal} \\\\\\\\\")")

    elif 'run_calibration_benchmark' in fp:
        if 'ACE' not in text:
            # Inject ACE calls
            text = text.replace("ece_raw = compute_ece(y_test, preds_raw)", "ece_raw = compute_ece(y_test, preds_raw)\n        ace_raw = compute_ace(y_test, preds_raw)")
            text = text.replace("'ECE': ece_raw,", "'ECE': ece_raw, 'ACE': ace_raw,")
            
            text = text.replace("ece_iso = compute_ece(y_test, preds_iso)", "ece_iso = compute_ece(y_test, preds_iso)\n            ace_iso = compute_ace(y_test, preds_iso)")
            text = text.replace("'ECE': ece_iso,", "'ECE': ece_iso, 'ACE': ace_iso,")
            
            text = text.replace("ece_platt = compute_ece(y_test, preds_platt)", "ece_platt = compute_ece(y_test, preds_platt)\n            ace_platt = compute_ace(y_test, preds_platt)")
            text = text.replace("'ECE': ece_platt,", "'ECE': ece_platt, 'ACE': ace_platt,")
            
            text = text.replace("ece_va = compute_ece(y_test, preds_va)", "ece_va = compute_ece(y_test, preds_va)\n                ace_va = compute_ace(y_test, preds_va)")
            text = text.replace("'ECE': ece_va,", "'ECE': ece_va, 'ACE': ace_va,")
            
            # Update tex lines
            text = text.replace(r"\textbf{ECE $\downarrow$} & \textbf{Brier $\downarrow$} \\", r"\textbf{ECE $\downarrow$} & \textbf{ACE $\downarrow$} & \textbf{Brier $\downarrow$} \\")
            text = text.replace(r"\begin{tabular}{lccc}", r"\begin{tabular}{lcccc}")
            text = text.replace("tex_lines.append(f\"{row['Model']} & {row['Calibrator']} & {pr} & {ece} & {brier} \\\\\\\\\")",
                                "ace = f\"{row.get('ACE', 0):.3f}\"\n        tex_lines.append(f\"{row['Model']} & {row['Calibrator']} & {pr} & {ece} & {ace} & {brier} \\\\\\\\\")")

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(text)

print("Master orchestrator patched with ACE columns!")

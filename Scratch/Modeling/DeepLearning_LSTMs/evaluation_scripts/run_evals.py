import subprocess
import sys

scripts = [
    ("CatBoost Non-Pipeline (Hurdle)", "train_catboost_survival.py", "petition_event"),
    ("CatBoost Non-Pipeline (Non-Hurdle)", "train_catboost_survival.py", "vote_event"),
    ("CatBoost Pipeline (Hurdle)", "train_hybrid_pipeline.py", "petition_event"),
    ("CatBoost Pipeline (Non-Hurdle)", "train_hybrid_pipeline.py", "vote_event"),
    ("LSTM Non-Causal (Hurdle)", "train_lstm_survival.py", "petition_event"),
    ("LSTM Non-Causal (Non-Hurdle)", "train_lstm_survival.py", "vote_event")
]

with open(r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts\evaluation_results.md", "w") as f:
    f.write("# Model Evaluations with Advanced Causal Features\n\n")

for name, script, target in scripts:
    print(f"\n=============================================\nRunning {name}...\n=============================================", flush=True)
    
    process = subprocess.Popen(
        f"python {script} {target}", 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        cwd=r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\DeepLearning_LSTMs"
    )
    
    report_lines = []
    capture = False
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip(), flush=True)
            if "ROC AUC:" in output or "Classification Report" in output or "Precision-Recall" in output:
                capture = True
            if capture:
                report_lines.append(output.strip())
                
    rc = process.poll()
    
    with open(r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts\evaluation_results.md", "a") as f:
        if rc == 0:
            report_text = "\n".join(report_lines)
            f.write(f"### {name}\n```\n{report_text}\n```\n")
        else:
            f.write(f"### {name}\n**FAILED**\n")

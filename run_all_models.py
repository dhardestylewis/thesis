import os
import time

commands = [
    (r"python Scratch\Modeling\DeepLearning_LSTMs\train_dynamic_lstm.py", "Dynamic LSTM"),
    (r"python Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\vae_gcomp_lstm_hybrid.py", "Causal Generative Model"),
    (r"python Scripts\pipeline\05_train_stage_c.py", "CatBoost Stage C"),
    (r"python Scripts\pipeline\08_run_meta_attribution.py", "SHAP Benchmarks")
]

print("Starting Sequential Training Run on Clean Leakage-Free Data...")
for cmd, name in commands:
    print(f"\n>>> Starting {name}...")
    start = time.time()
    os.system(cmd)
    end = time.time()
    print(f"<<< Finished {name} in {(end-start)/60:.1f} minutes.")

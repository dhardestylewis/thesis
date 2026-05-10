import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from run_causal_flow_transformer import load_and_prep_data, extract_tensors, CausalFlowTransformer, rk4_inpaint_solver
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def main():
    print("[*] Loading Data...")
    df, features, norm_dict = load_and_prep_data()
    
    # Split into train/test
    cases = df['case_number'].unique()
    np.random.seed(42)
    np.random.shuffle(cases)
    
    train_cases = cases[:int(len(cases)*0.8)]
    test_cases = cases[int(len(cases)*0.8):]
    
    df_train = df[df['case_number'].isin(train_cases)]
    df_test = df[df['case_number'].isin(test_cases)]
    
    X_train = extract_tensors(df_train, features, max_len=30)
    X_test = extract_tensors(df_test, features, max_len=30)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    
    model = CausalFlowTransformer(feature_dim=len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
    
    print("[*] Training CFM-T and tracking metrics over 150 Epochs...")
    EPOCHS_LIST = [20, 50, 100, 150]
    current_epoch = 0
    
    for target_epochs in EPOCHS_LIST:
        model.train()
        while current_epoch < target_epochs:
            for batch in dataloader:
                x_1 = batch[0]
                B = x_1.size(0)
                optimizer.zero_grad()
                x_0 = torch.randn_like(x_1)
                t = torch.rand(B, device=device)
                t_expand = t.view(B, 1, 1)
                x_t = (1 - t_expand) * x_0 + t_expand * x_1
                v_target = x_1 - x_0
                v_pred = model(x_t, t)
                loss = nn.MSELoss()(v_pred, v_target)
                loss.backward()
                optimizer.step()
            current_epoch += 1
            
        print(f"\n[*] Evaluating Metrics at Epoch {current_epoch}...")
        model.eval()
        with torch.no_grad():
            x_0 = torch.randn_like(X_test)
            x_t = x_0
            steps = 10
            dt = 1.0 / steps
            for step in range(steps):
                t_val = torch.ones(X_test.size(0), device=device) * (step * dt)
                v = model(x_t, t_val)
                x_t = x_t + v * dt
            
            generated_unconditional = x_t.cpu().numpy()
            real_unconditional = X_test.cpu().numpy()
            var_gen = np.var(generated_unconditional)
            var_real = np.var(real_unconditional)
            
            intervention_p = 5
            generated_conditional = x_0.clone()
            for step in range(steps):
                t_val = torch.ones(X_test.size(0), device=device) * (step * dt)
                generated_conditional[:, :intervention_p, :] = X_test[:, :intervention_p, :]
                v = model(generated_conditional, t_val)
                generated_conditional = generated_conditional + v * dt
                
            final_gen = generated_conditional.cpu().numpy()
            idx_dg = features.index("t_downgrade")
            
            y_probs = final_gen[:, -1, idx_dg]
            y_probs = np.clip(y_probs, 0, 1)
            y_true = real_unconditional[:, -1, idx_dg]
            
            prauc = average_precision_score(y_true, y_probs)
            baseline = np.mean(y_true)
            
            print(f"  > Variance -> Real: {var_real:.4f} | Generated: {var_gen:.4f}")
            print(f"  > PRAUC -> CFM-T: {prauc:.4f} | Baseline: {baseline:.4f}")

if __name__ == "__main__":
    main()

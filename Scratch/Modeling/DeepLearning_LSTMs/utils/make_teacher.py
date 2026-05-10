with open(r'c:\Users\dhl\data\Thesis\thesis\Scratch\train_gru_survival.py', 'r') as f:
    code = f.read()

code += '''
df["raw_period_seq"] = df_raw["period_seq"] if "df_raw" in locals() else df["period_seq"]
print("\\n5. Saving GRU Soft Probabilities for Knowledge Distillation...")
all_dataset = ZoningHazardDataset(df)
all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

case_seq_probs = []

model.eval()
with torch.no_grad():
    for batch_idx, (x, y, mask) in enumerate(all_loader):
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        start_idx = batch_idx * 64
        for i in range(len(x)):
            case_idx = start_idx + i
            if case_idx >= len(all_dataset.groups): break
            case_num, group = all_dataset.groups[case_idx]
            group = group.sort_values("period_seq")
            
            seq_len = int(mask[i].sum().item())
            
            for t in range(seq_len):
                period = group.iloc[t]["raw_period_seq"]
                p_val = probs[i, t, 0]
                case_seq_probs.append({
                    "case_number": case_num,
                    "period_seq": period,
                    "gru_prob": p_val
                })

prob_df = pd.DataFrame(case_seq_probs)
out_path = f"C:\\\\Users\\\\dhl\\\\data\\\\Thesis\\\\thesis\\\\Data\\\\Panel\\\\gru_probs_{TARGET}.csv"
prob_df.to_csv(out_path, index=False)
print(f"Saved {len(prob_df)} soft probabilities to {out_path}")
'''

with open(r'c:\Users\dhl\data\Thesis\thesis\Scratch\train_gru_teacher.py', 'w') as f:
    f.write(code)

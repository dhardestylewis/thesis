import numpy as np
import matplotlib.pyplot as plt
import os

print("Rendering F16: Petition Threshold RD Plot...")
out_dir = r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Figures\Chapter5"
os.makedirs(out_dir, exist_ok=True)

# Generate synthetic Running Variable (Signed Area Share)
np.random.seed(42)
running_var_left = np.random.uniform(0, 0.199, 300)
running_var_right = np.random.uniform(0.20, 0.50, 100)

# Simulate continuous outcome (e.g., Delay / Continuance Probability)
outcome_left = 0.15 + 0.5 * running_var_left + np.random.normal(0, 0.05, 300)
# The Treatment Effect (Jump at 20%)
outcome_right = 0.45 + 0.2 * running_var_right + np.random.normal(0, 0.08, 100)

plt.figure(figsize=(9, 6))
plt.scatter(running_var_left, outcome_left, alpha=0.3, color='gray', s=15, label='Control (Valid Petition < 20%)')
plt.scatter(running_var_right, outcome_right, alpha=0.5, color='darkred', s=15, label=r'Treated (Valid Petition $\geq$ 20%)')

# Fit polynomials
x_left = np.linspace(0, 0.20, 100)
y_left = 0.15 + 0.5 * x_left
x_right = np.linspace(0.20, 0.50, 100)
y_right = 0.45 + 0.2 * x_right

plt.plot(x_left, y_left, color='black', linewidth=2.5)
plt.plot(x_right, y_right, color='black', linewidth=2.5)

plt.axvline(x=0.20, color='red', linestyle='--', linewidth=2, label='Statutory 20% Threshold')
plt.title('Exhibit F16: Regression Discontinuity around 20% Protest Threshold', fontsize=14, pad=15)
plt.xlabel('Valid Protest Petition Signed Area Share', fontsize=12)
plt.ylabel('Probability of Substantial Council Delay / Continuance', fontsize=12)
plt.legend(loc='upper left', fontsize=11, frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

f16_path = os.path.join(out_dir, "F16_Petition_RD.png")
plt.savefig(f16_path, dpi=300)
print(f"Successfully saved {f16_path}")

import os
import subprocess

scripts_to_run = [
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Warehouse_Builder\14_generate_visualizations.py",
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Warehouse_Builder\33_generate_track1_visualizations.py",
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Experiments\DiD\generate_thesis_figures.py",
    # Modeling benchmarks that generate LaTeX tables
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Modeling\Production_Models\run_alternative_architectures.py",
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Modeling\Production_Models\run_calibration_benchmark.py",
]

# Add all Visualization scripts
viz_dir = r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Visualization"
for root, _, files in os.walk(viz_dir):
    for f in files:
        if f.startswith("plot_") and f.endswith(".py"):
            scripts_to_run.append(os.path.join(root, f))

for script in scripts_to_run:
    print(f"Running {os.path.basename(script)}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts"
    try:
        subprocess.run(["python", script], check=True, cwd=os.path.dirname(script), env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")

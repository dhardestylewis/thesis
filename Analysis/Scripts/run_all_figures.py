import os
import subprocess

scripts_to_run = [
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Warehouse_Builder\14_generate_visualizations.py",
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Warehouse_Builder\33_generate_track1_visualizations.py",
    r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Experiments\DiD\generate_thesis_figures.py",
]

# Add all Visualization scripts
viz_dir = r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts\Visualization"
for f in os.listdir(viz_dir):
    if f.startswith("plot_") and f.endswith(".py"):
        scripts_to_run.append(os.path.join(viz_dir, f))

for script in scripts_to_run:
    print(f"Running {os.path.basename(script)}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts"
    try:
        subprocess.run(["python", script], check=True, cwd=os.path.dirname(script), env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")

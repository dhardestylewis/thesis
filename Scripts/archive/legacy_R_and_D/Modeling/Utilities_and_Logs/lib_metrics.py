import os
import re

ROOT = r"C:\Users\dhl\data\thesis\thesis"
METRICS_FILE = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "metrics_config.tex")

def update_metric(macro_name, new_value):
    """
    Safely updates or adds a specific LaTeX macro in metrics_config.tex.
    This prevents wiping out existing variables if a single script fails to run.
    """
    if not os.path.exists(METRICS_FILE):
        content = ""
    else:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            content = f.read()

    # Check if the macro already exists
    pattern = r"\\newcommand\{\\" + macro_name + r"\}\{.*?\}"
    replacement = f"\\newcommand{{\\{macro_name}}}{{{new_value}}}"

    if re.search(pattern, content):
        # Overwrite the existing definition (using lambda to prevent escape parsing errors)
        content = re.sub(pattern, lambda _: replacement, content)
    else:
        # Append the new definition
        content += f"\n{replacement}\n"

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(content)

def get_current_metrics():
    """Returns a dictionary of current macros."""
    # (Utility function to inspect state if needed)
    pass

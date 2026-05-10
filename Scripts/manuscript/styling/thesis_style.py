import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# Columbia University + Okabe-Ito Academic Palette
COLUMBIA_BLUE = "#B9D9EB"
DARK_NAVY = "#002B7F" # Columbia standard navy
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

def set_thesis_style():
    """
    Applies a centralized, academically rigorous, and colorblind-friendly
    plotting style to matplotlib and seaborn.
    Call this function at the start of any visualization script.
    """
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.edgecolor": "#333333",
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            "font.family": "serif",        # Matches LaTeX document
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    
    # Set the default color cycle to Okabe-Ito
    plt.rc('axes', prop_cycle=cycler('color', OKABE_ITO))
    
    # Optional: ensure correct DPI for high-quality PNGs
    plt.rc('figure', dpi=300)
    plt.rc('savefig', dpi=300, bbox='tight')

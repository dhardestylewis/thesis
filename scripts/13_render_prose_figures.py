import sys
import os
from pathlib import Path
sys.path.append(r'c:\Users\dhl\data\thesis\thesis')
import os, glob; [os.system(f'python "{f}"') for f in glob.glob('Analysis/Scripts/Visualization/Production_Figures/*.py') + glob.glob('Analysis/Scripts/Experiments/DiD/*.py') if 'electoral_placebo' in f or 'generate_' in f or 'plot_' in f]

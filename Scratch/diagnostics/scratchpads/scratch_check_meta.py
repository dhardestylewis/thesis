import pandas as pd, numpy as np, os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA_DIR = os.path.join(ROOT, 'Analysis', 'Output', 'SHAP_MetaClustering')

# Let's see what files are in the SHAP_MetaClustering folder
print("Files in SHAP_MetaClustering:")
print(os.listdir(DATA_DIR))


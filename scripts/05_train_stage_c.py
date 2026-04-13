import sys
import os
from pathlib import Path
sys.path.append(r'c:\Users\dhl\data\thesis\thesis')
from src.models.train_stage_c import train_stage_c; train_stage_c('CatBoost')

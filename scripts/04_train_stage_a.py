import sys
import os
from pathlib import Path
sys.path.append(r'c:\Users\dhl\data\thesis\thesis')
from src.features.build_stage_a_features import build_stage_a_features; from src.models.train_stage_a import train_stage_a_ipw; build_stage_a_features(); train_stage_a_ipw()

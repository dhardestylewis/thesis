import sys
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Modeling"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Visualization"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Experiments"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Experiments", "DiD"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Warehouse_Builder"))

import plot_F8_Calibration_real as f8
import plot_F12_PR_real as f12
import plot_F16_RD_real as f16
import plot_F17_DiD_real as f17
import plot_Track1_exhibits_real as t1_ex
import plot_F19_F20_Qualitative as f19_f20
import plot_F22_HexMap as f22
import generate_stageA_exhibits as stage_a_exhibits
import generate_thesis_figures as fig1_and_more
import importlib
try:
    sweeps = importlib.import_module("18_real_model_sweeps")
except Exception as e:
    sweeps = None

print("Regenerating all visualizations with unified style...")

try: 
    f8.plot_f8()
    print("Done F8")
except Exception as e: print("Error F8:", e)

try: 
    f12.plot_f12()
    print("Done F12")
except Exception as e: print("Error F12:", e)

try: 
    f16.plot_f16()
    print("Done F16")
except Exception as e: print("Error F16:", e)

try: 
    f17.plot_f17()
    print("Done F17")
except Exception as e: print("Error F17:", e)

try: 
    t1_ex.plot_all_track1_exhibits()
    print("Done Track 1 exhibits")
except Exception as e: print("Error Track 1:", e)

try: 
    f19_f20.generate_exhibits()
    print("Done F19/20")
except Exception as e: print("Error F19/20:", e)

try: 
    f22.generate_exhibits()
    print("Done F22")
except Exception as e: print("Error F22:", e)

try: 
    stage_a_exhibits.generate_exhibits()
    print("Done Stage A")
except Exception as e: print("Error Stage A:", e)

try: 
    fig1_and_more.main()
    print("Done Fig1")
except Exception as e: print("Error Fig1:", e)

try:
    if sweeps: 
        sweeps.run_real_pipelines()
    print("Done sweeps")
except Exception as e: print("Error sweeps:", e)

print("All regenerations complete!")

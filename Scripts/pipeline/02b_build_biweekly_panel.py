import os
import pandas as pd
import shutil

def main():
    SOURCE = r"c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv"
    DEST = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
    
    print(f"Step 12: Copying fully hydrated biweekly panel to causal modeling directory...")
    print(f"Source: {SOURCE}")
    print(f"Dest:   {DEST}")
    
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    shutil.copy2(SOURCE, DEST)
    
    sz = os.path.getsize(DEST) / 1e6
    print(f"\nSuccessfully deployed {sz:.1f} MB formal causal inference panel.")

if __name__ == "__main__":
    main()

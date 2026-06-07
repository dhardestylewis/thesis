import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
from tqdm import tqdm

sys.path.append(r"c:\Users\dhl\data\thesis\thesis\Scripts\pipeline")
from execute_thesis_pipeline import PIPELINE_STEPS, run_step, load_timings, DEFAULT_TIMING, SCRIPTS_DIR

def main():
    os.environ["USE_GPU"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Start from index
    start_idx = 0
    for i, (s, d) in enumerate(PIPELINE_STEPS):
        if s == "12_generate_extracted_tables.py":
            start_idx = i
            break
            
    remaining_steps = PIPELINE_STEPS[start_idx:]
    
    timings = load_timings()
    total_expected = sum([timings.get(s[0], DEFAULT_TIMING) for s in remaining_steps])
    
    print(f"Resuming pipeline from script: {remaining_steps[0][0]}")
    
    with tqdm(total=total_expected, desc="Pipeline ETA Progress", unit="s", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for script, desc in remaining_steps:
            expected = timings.get(script, DEFAULT_TIMING)
            run_step(script, desc, pbar, expected)

if __name__ == "__main__":
    main()

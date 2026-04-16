import os
import pandas as pd
import subprocess
import time
import argparse
import json
from faster_whisper import WhisperModel

# Setup paths
DATA_DIR = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data"
QUEUE_CSV = os.path.join(DATA_DIR, "transcription_queue_full.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "Transcripts")

def main():
    parser = argparse.ArgumentParser(description="Transcribe Austin Zoning videos via local faster-whisper.")
    parser.add_argument("--sample", type=int, help="Number of random cases to sample and transcribe", default=None)
    args = parser.parse_args()

    if not os.path.exists(QUEUE_CSV):
        print(f"Error: {QUEUE_CSV} not found.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading faster-whisper model (small.en)...")
    try:
        model = WhisperModel("small.en", device="cuda", compute_type="float16")
        print("✓ Model loaded on CUDA (GPU).")
    except Exception as e:
        print(f"CUDA load failed: {e}. Falling back to CPU with int8 (this will be slower).")
        model = WhisperModel("small.en", device="cpu", compute_type="int8")

    df = pd.read_csv(QUEUE_CSV)
    
    # Filter to only the target multi-parcel cases
    TARGET_CSV = os.path.join(DATA_DIR, "multi_parcel_closed_2018_2025.csv")
    if os.path.exists(TARGET_CSV):
        target_cases = pd.read_csv(TARGET_CSV)['CASE_NUMBER'].unique()
        df = df[df['CASE_NUMBER'].isin(target_cases)]
    
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Randomly sampled {len(df)} records for testing.")
        
    print(f"Transcription queue (filtered): {len(df)} files.")
    
    # Track completion
    completed_csv = os.path.join(DATA_DIR, "transcription_status_full.csv")
    if os.path.exists(completed_csv):
        status_df = pd.read_csv(completed_csv)
    else:
        status_df = pd.DataFrame(columns=["CASE_NUMBER", "Status", "Message"])
        
    completed_cases = set(status_df[status_df['Status'] == 'Success']['CASE_NUMBER'].tolist())
    
    results = status_df.to_dict('records')
    
    for idx, row in df.iterrows():
        case = row['CASE_NUMBER']
        mp4_url = row['MP4_URL']
        
        if case in completed_cases:
            continue
            
        print(f"\nProcessing [{idx+1}/{len(df)}]: {case}")
        transcript_path = os.path.join(OUTPUT_DIR, f"{case.replace('/', '_')}_transcript.txt")
        mp3_path = os.path.join(OUTPUT_DIR, f"temp_{case.replace('/', '_')}.mp3")
        
        # 1. Download audio/video via HTTP
        mp4_path = os.path.join(OUTPUT_DIR, f"temp_{case.replace('/', '_')}.mp4")
        print(f"  -> Downloading video from {mp4_url}...")
        
        try:
            import requests
            response = requests.get(mp4_url, stream=True, timeout=60)
            response.raise_for_status()
            with open(mp4_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            msg = f"Download failed: {str(e)}"
            print(f"  -> {msg}")
            results.append({"CASE_NUMBER": case, "Status": "Failed", "Message": msg})
            continue
            
        # 2. Transcribe via local faster-whisper
        print("  -> Transcribing locally using faster-whisper...")
        
        if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) == 0:
            msg = "Downloaded file is empty."
            print(f"  -> {msg}")
            results.append({"CASE_NUMBER": case, "Status": "Failed", "Message": msg})
            continue

        try:
            segments, info = model.transcribe(mp4_path, beam_size=5)
            
            # Combine all segments into full text
            full_text = []
            for segment in segments:
                full_text.append(segment.text)
            
            transcription = " ".join(full_text)
                
            with open(transcript_path, "w", encoding="utf-8") as text_file:
                # Add header metadata
                text_file.write(f"Zoning Case: {case}\n")
                text_file.write(f"Meeting Date: {row['Meeting_Date']}\n")
                text_file.write(f"Agenda Item: {row['Agenda_Item']}\n")
                text_file.write(f"Source URL: {row['Swagit_URL']}\n")
                text_file.write("="*80 + "\n\n")
                text_file.write(transcription)
                
            print(f"  -> Success! Transcript saved to {transcript_path}")
            results.append({"CASE_NUMBER": case, "Status": "Success", "Message": "Completed"})
            
        except Exception as e:
            msg = f"Transcription error: {str(e)}"
            print(f"  -> {msg}")
            results.append({"CASE_NUMBER": case, "Status": "Failed", "Message": msg})
            
        finally:
            # Cleanup temp mp4
            if os.path.exists(mp4_path):
                try:
                    os.remove(mp4_path)
                except:
                    pass
                
        # Save progress after every file
        pd.DataFrame(results).to_csv(completed_csv, index=False)

if __name__ == "__main__":
    main()

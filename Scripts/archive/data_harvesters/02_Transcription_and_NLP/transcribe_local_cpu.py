import os
import pandas as pd
import subprocess
import time
from faster_whisper import WhisperModel
import imageio_ffmpeg

# Setup paths
DATA_DIR = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data"
QUEUE_CSV = os.path.join(DATA_DIR, "CSV", "transcription_queue_full.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "Transcripts")

def main():
    if not os.path.exists(QUEUE_CSV):
        print(f"Error: {QUEUE_CSV} not found.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the very fast/small Whisper model optimized for CPU
    print("Loading faster-whisper 'base' model for CPU computation...")
    # int8 quantization vastly reduces memory and speeds up CPU inference
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    
    df = pd.read_csv(QUEUE_CSV)
    print(f"Loaded queue with {len(df)} total items.")
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    for idx, row in df.iterrows():
        case = row['CASE_NUMBER']
        mp4_url = row['MP4_URL']
        
        # 1. Resiliency check: If the final transcript exists, we skip immediately.
        # This handles the "ephemeral" laptop (closing lid, crashing, manual stopping).
        transcript_path = os.path.join(OUTPUT_DIR, f"{case.replace('/', '_')}_transcript.txt")
        if os.path.exists(transcript_path):
            continue
            
        print(f"\n[{idx+1}/{len(df)}] Processing Case: {case}")
        mp3_path = os.path.join(OUTPUT_DIR, f"temp_{case.replace('/', '_')}.mp3")
        
        # 2. Extract Audio
        print(f"  -> Extracting audio via FFmpeg...")
        cmd = [
            ffmpeg_exe, "-y", "-i", mp4_url, 
            "-vn", "-acodec", "libmp3lame", "-q:a", "8", 
            mp3_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600, check=True)
        except Exception as e:
            print(f"  -> FFmpeg error or timeout: {e}")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            continue
            
        # 3. Transcribe via Local CPU
        print("  -> Transcribing using local CPU (this may take a few minutes)...")
        try:
            segments, info = model.transcribe(mp3_path, beam_size=5)
            
            # Write to a temporary file first, then rename it, so we don't 
            # create a corrupted/half-finished transcript if the laptop closes.
            temp_transcript = transcript_path + ".tmp"
            
            with open(temp_transcript, "w", encoding="utf-8") as f:
                f.write(f"Zoning Case: {case}\n")
                f.write(f"Meeting Date: {row['Meeting_Date']}\n")
                f.write(f"Agenda Item: {row['Agenda_Item']}\n")
                f.write(f"Source URL: {row['Swagit_URL']}\n")
                f.write("="*80 + "\n\n")
                
                # As the model processes chunks of audio, write them to disk immediately.
                for segment in segments:
                    f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
            
            # Atomic rename ensures the file is fully baked
            os.replace(temp_transcript, transcript_path)
            print(f"  -> Success! Saved to {transcript_path}")
            
        except Exception as e:
            print(f"  -> Transcription error: {str(e)}")
            if os.path.exists(temp_transcript):
                os.remove(temp_transcript)
        finally:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

if __name__ == "__main__":
    main()

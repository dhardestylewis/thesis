"""
transcribe_modal.py — GPU-Accelerated Batch Transcription via Modal
====================================================================
Processes all 3,241 Austin zoning case videos in parallel on cloud GPUs.
Each video is downloaded, audio-extracted, and transcribed inside a
serverless container. Results are saved back to local disk.

Usage:
  modal run transcribe_modal.py           # Process all items
  modal run transcribe_modal.py --sample 344  # Statistically significant sample

Cost estimate: ~$0.50-$2.00 for the full 3,241 items on T4 GPUs.
"""

import modal
import os

app = modal.App("austin-zoning-transcripts")

# Container image with all dependencies baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper", "pandas")
)

# Persistent volume to store results across runs (resumable)
vol = modal.Volume.from_name("zoning-transcripts", create_if_missing=True)
REMOTE_OUTPUT_DIR = "/results"


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    retries=2,
    volumes={REMOTE_OUTPUT_DIR: vol},
)
def transcribe_one(case_number: str, mp4_url: str, meeting_date: str,
                   agenda_item: str, swagit_url: str) -> str:
    """Transcribe a single video inside a GPU container."""
    import subprocess
    import tempfile
    from faster_whisper import WhisperModel

    safe_name = case_number.replace("/", "_")
    output_path = f"{REMOTE_OUTPUT_DIR}/{safe_name}_transcript.txt"

    # Resumability: skip if already done
    if os.path.exists(output_path):
        return f"SKIP: {case_number} (already transcribed)"

    # 1. Download audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp4_url, "-vn", "-acodec", "libmp3lame",
             "-q:a", "8", mp3_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=300, check=True
        )
    except Exception as e:
        return f"FAIL (ffmpeg): {case_number} — {e}"

    # 2. Transcribe on GPU
    try:
        model = WhisperModel("base.en", device="cuda", compute_type="float16")
        segments, _ = model.transcribe(mp3_path, beam_size=5)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Zoning Case: {case_number}\n")
            f.write(f"Meeting Date: {meeting_date}\n")
            f.write(f"Agenda Item: {agenda_item}\n")
            f.write(f"Source URL: {swagit_url}\n")
            f.write("=" * 80 + "\n\n")
            for seg in segments:
                f.write(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}\n")

        vol.commit()
        return f"OK: {case_number}"
    except Exception as e:
        return f"FAIL (whisper): {case_number} — {e}"
    finally:
        os.remove(mp3_path)


@app.local_entrypoint()
def main(sample: int = 0):
    """Launch parallel transcription across all queued videos."""
    import pandas as pd

    queue_csv = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\transcription_queue_full.csv"
    df = pd.read_csv(queue_csv)

    if sample > 0:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        print(f"Randomly sampled {len(df)} items.")

    print(f"Launching {len(df)} transcription jobs on Modal GPUs...")

    # .map() fans out across many containers in parallel
    results = list(transcribe_one.map(
        df["CASE_NUMBER"].tolist(),
        df["MP4_URL"].tolist(),
        df["Meeting_Date"].astype(str).tolist(),
        df["Agenda_Item"].astype(str).tolist(),
        df["Swagit_URL"].tolist(),
    ))

    ok = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    fail = sum(1 for r in results if r.startswith("FAIL"))
    print(f"\nDone! OK={ok}, Skipped={skip}, Failed={fail}")

    # Download results from the Modal volume to local disk
    local_dir = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\Transcripts"
    os.makedirs(local_dir, exist_ok=True)

    print(f"Syncing transcripts from Modal volume to {local_dir}...")
    for entry in vol.listdir("/"):
        if entry.path.endswith("_transcript.txt"):
            local_path = os.path.join(local_dir, os.path.basename(entry.path))
            if not os.path.exists(local_path):
                with open(local_path, "wb") as f:
                    for chunk in vol.read_file(entry.path):
                        f.write(chunk)

    print("Sync complete!")

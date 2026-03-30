"""
transcribe_aws.py — GPU-Accelerated Batch Transcription on AWS EC2 Spot
========================================================================
Designed to run on a cheap GPU spot instance (e.g., g4dn.xlarge ~$0.16/hr).
Downloads videos, transcribes with faster-whisper, and uploads results to S3.

Setup (one-time on instance):
  pip install faster-whisper pandas boto3
  sudo apt install ffmpeg

Usage:
  # Upload queue CSV to S3 first:
  aws s3 cp transcription_queue_full.csv s3://YOUR_BUCKET/zoning/queue.csv

  # SSH into instance and run:
  python transcribe_aws.py --bucket YOUR_BUCKET --sample 344

  # Or process everything:
  python transcribe_aws.py --bucket YOUR_BUCKET

Cost estimate: g4dn.xlarge spot = ~$0.16/hr. Full 3,241 items in ~2 hrs = ~$0.32.
"""

import os
import argparse
import subprocess
import tempfile
import time

import boto3
import pandas as pd
from faster_whisper import WhisperModel

S3_PREFIX = "zoning/transcripts/"


def get_completed_keys(s3, bucket):
    """List all already-completed transcripts in S3 for resumability."""
    completed = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            basename = os.path.basename(key)
            if basename.endswith("_transcript.txt"):
                case = basename.replace("_transcript.txt", "")
                completed.add(case)
    return completed


def transcribe_one(model, row, s3, bucket):
    """Download, extract audio, transcribe, upload to S3."""
    case = row["CASE_NUMBER"]
    mp4_url = row["MP4_URL"]
    safe_name = case.replace("/", "_")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = tmp.name

    # 1. Extract audio via FFmpeg (streams directly from URL)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp4_url, "-vn", "-acodec", "libmp3lame",
             "-q:a", "8", mp3_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=300, check=True
        )
    except Exception as e:
        print(f"  FAIL (ffmpeg): {case} — {e}")
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return False

    # 2. Transcribe on GPU
    try:
        segments, _ = model.transcribe(mp3_path, beam_size=5)

        # Build transcript text
        lines = [
            f"Zoning Case: {case}",
            f"Meeting Date: {row['Meeting_Date']}",
            f"Agenda Item: {row['Agenda_Item']}",
            f"Source URL: {row['Swagit_URL']}",
            "=" * 80, "",
        ]
        for seg in segments:
            lines.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")

        transcript_text = "\n".join(lines)

        # 3. Upload to S3
        s3_key = f"{S3_PREFIX}{safe_name}_transcript.txt"
        s3.put_object(Bucket=bucket, Key=s3_key, Body=transcript_text.encode("utf-8"))
        print(f"  OK: {case} -> s3://{bucket}/{s3_key}")
        return True

    except Exception as e:
        print(f"  FAIL (whisper): {case} — {e}")
        return False
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe zoning videos on AWS EC2 GPU.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--queue", default="transcription_queue_full.csv",
                        help="Local path or S3 key (s3://bucket/key) to the queue CSV")
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of random items to sample (0 = all)")
    args = parser.parse_args()

    s3 = boto3.client("s3")

    # Load queue (from local file or S3)
    if args.queue.startswith("s3://"):
        parts = args.queue.replace("s3://", "").split("/", 1)
        local_queue = "/tmp/queue.csv"
        s3.download_file(parts[0], parts[1], local_queue)
        df = pd.read_csv(local_queue)
    else:
        df = pd.read_csv(args.queue)

    if args.sample > 0:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Randomly sampled {len(df)} items.")

    # Check what's already done (resumability)
    print("Checking S3 for previously completed transcripts...")
    completed = get_completed_keys(s3, args.bucket)
    remaining = df[~df["CASE_NUMBER"].apply(lambda c: c.replace("/", "_")).isin(completed)]
    print(f"Total: {len(df)}, Already done: {len(df) - len(remaining)}, Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All items already transcribed!")
        return

    # Load model once (reuse across all items)
    print("Loading faster-whisper model on GPU...")
    model = WhisperModel("base.en", device="cuda", compute_type="float16")

    ok, fail = 0, 0
    t0 = time.time()
    for i, (_, row) in enumerate(remaining.iterrows()):
        print(f"\n[{i+1}/{len(remaining)}] Case: {row['CASE_NUMBER']}")
        if transcribe_one(model, row, s3, args.bucket):
            ok += 1
        else:
            fail += 1

        # Print ETA every 10 items
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            per_item = elapsed / (i + 1)
            eta_min = per_item * (len(remaining) - i - 1) / 60
            print(f"  [{i+1}/{len(remaining)}] avg={per_item:.1f}s/item, ETA={eta_min:.0f} min")

    print(f"\nDone! OK={ok}, Failed={fail}")
    print(f"Results in: s3://{args.bucket}/{S3_PREFIX}")


if __name__ == "__main__":
    main()

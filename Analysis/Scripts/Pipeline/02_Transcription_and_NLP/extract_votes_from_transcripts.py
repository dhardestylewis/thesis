import os
import pandas as pd
import json
import time
import requests
import argparse

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_DIR, "Data", "Zoning_Cases", "Processed_Data")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "Transcripts")
OUTPUT_CSV = os.path.join(DATA_DIR, "transcript_votes_parsed.csv")

def parse_transcript_metadata(transcript_text):
    """Extract metadata from the transcript file header."""
    metadata = {}
    lines = transcript_text.split('\n')
    for line in lines[:10]:
        if line.startswith("Zoning Case:"):
            metadata['CASE_NUMBER'] = line.replace("Zoning Case:", "").strip()
        elif line.startswith("Meeting Date:"):
            metadata['Meeting_Date'] = line.replace("Meeting Date:", "").strip()
        elif line.startswith("Agenda Item:"):
            metadata['Agenda_Item'] = line.replace("Agenda Item:", "").strip()
        elif line.startswith("Source URL:"):
            metadata['Source_URL'] = line.replace("Source URL:", "").strip()
        elif line.startswith("====="):
            break
    return metadata

def extract_vote_from_text(text, case_number, model="llama3"):
    """Uses Ollama local API with JSON format to extract vote data."""
    system_prompt = """You are an expert transcriber of Austin City Council meetings.
Your task is to identify and extract the final vote outcome for a particular zoning case.
You will receive the transcript of an agenda item.
Focus specifically on the final roll call or voice vote outcome declared by the Mayor or clerk (e.g., "That passes on a vote of 10 to 1...").
If the item was postponed or no vote took place, return null for vote_yes and vote_no.

You must reply with ONLY a valid JSON object matching this exact schema:
{
  "vote_yes": integer or null,
  "vote_no": integer or null,
  "reading": string ("1st", "2nd", "3rd", "2nd/3rd", or "unknown"),
  "nay_members": string (comma-separated list of members who voted against, empty if none),
  "absent_members": string (comma-separated list of members absent/off dais, empty if none),
  "confidence": string ("High", "Medium", or "Low"),
  "reasoning": string (short sentence explaining how you derived the vote)
}
"""
    
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract the vote outcome for case {case_number}:\n\n{text}"}
        ],
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        result_json = response.json()["message"]["content"]
        # Parse the returned JSON
        return json.loads(result_json)
    except Exception as e:
        print(f"Error during Ollama API call: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract votes from transcripts via local Ollama.")
    parser.add_argument("--model", type=str, help="Ollama model to use", default="llama3")
    args = parser.parse_args()

    # Check if Ollama is running
    try:
        requests.get("http://localhost:11434/", timeout=2)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama at http://localhost:11434/. Please ensure Ollama is running.")
        return

    print(f"Using local Ollama model: {args.model}")

    if not os.path.exists(TRANSCRIPTS_DIR):
        print(f"Directory not found: {TRANSCRIPTS_DIR}")
        return

    # Load existing results to allow resuming
    if os.path.exists(OUTPUT_CSV):
        out_df = pd.read_csv(OUTPUT_CSV)
        processed_cases = set(out_df['CASE_NUMBER'].tolist())
    else:
        out_df = pd.DataFrame()
        processed_cases = set()

    results = []
    
    files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith("_transcript.txt")][:5]
    print(f"Found {len(files)} transcripts to process.")
    
    for idx, filename in enumerate(files):
        print(f"[{idx+1}/{len(files)}] Processing {filename}...")
        
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        metadata = parse_transcript_metadata(text)
        case_number = metadata.get('CASE_NUMBER', filename.replace('_transcript.txt', ''))
        
        if case_number in processed_cases:
            print(f"  -> Already processed {case_number}, skipping.")
            continue
            
        transcript_body = text.split("================================================================================")[-1]
        
        # Ollama local models have smaller context windows (usually 8k).
        # To be safe, we might truncate to the last 15000 chars since votes are at the end,
        # but modern local models (like Llama 3) support 8k tokens (~32k chars). Let's pass the whole thing.
        
        vote_data = extract_vote_from_text(transcript_body, case_number, model=args.model)
        
        if vote_data:
            results.append({
                "CASE_NUMBER": case_number,
                "Meeting_Date": metadata.get("Meeting_Date", ""),
                "Agenda_Item": metadata.get("Agenda_Item", ""),
                "vote_yes": vote_data.get("vote_yes"),
                "vote_no": vote_data.get("vote_no"),
                "reading_stage": vote_data.get("reading", "unknown"),
                "nay_members": vote_data.get("nay_members", ""),
                "absent_members": vote_data.get("absent_members", ""),
                "confidence": vote_data.get("confidence", "Low"),
                "reasoning": vote_data.get("reasoning", ""),
                "Source_URL": metadata.get("Source_URL", "")
            })
            print(f"  -> Extracted: Yes: {vote_data.get('vote_yes', '?')} - No: {vote_data.get('vote_no', '?')}")
        
        time.sleep(0.1) # Small pause
        
        # Save every 5 records to avoid losing progress
        if len(results) > 0 and len(results) % 5 == 0:
            temp_df = pd.concat([out_df, pd.DataFrame(results)], ignore_index=True)
            temp_df.to_csv(OUTPUT_CSV, index=False)
            out_df = temp_df
            results = []

    # Final save
    if len(results) > 0:
        temp_df = pd.concat([out_df, pd.DataFrame(results)], ignore_index=True)
        temp_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(temp_df)} total records to {OUTPUT_CSV}")
    else:
        print("Done.")

if __name__ == "__main__":
    main()

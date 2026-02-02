import os
import time
import random
import json
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Activity Log Path
LOG_FILE = "ACTIVITY_LOG.json"

# Scopes required for sending
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose', 
    'https://www.googleapis.com/auth/gmail.send' 
]

def log_sent_activity(draft_id, message_id, recipient):
    """Logs the sending activity to JSON file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "sent_email",
        "draft_id": draft_id,
        "message_id": message_id,
        "recipient": recipient
    }
    
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                content = json.load(f)
                if isinstance(content, dict):
                    if "activities" not in content:
                        content["activities"] = []
                    content["activities"].append(entry)
                else:
                    content.append(entry)
        except json.JSONDecodeError:
            content = [entry]
    else:
        content = [entry]
        
    with open(LOG_FILE, 'w') as f:
        json.dump(content, f, indent=2)

def get_credentials():
    """Gets valid user credentials."""
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        
        if not creds:
            print("Error: Credentials invalid or missing. Run create_mass_drafts.py first to auth.")
            return None
            
    return creds

def send_drafts():
    """Sends the outreach drafts logged in ACTIVITY_LOG.json."""
    creds = get_credentials()
    if not creds:
        return

    service = build('gmail', 'v1', credentials=creds)
    
    # Identify drafts to send from the log
    drafts_to_send = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                content = json.load(f)
                logs = content.get("activities", []) if isinstance(content, dict) else content
                
                # We only want to send drafts to recipients that haven't been sent to yet
                sent_recipients = {l.get("recipient") for l in logs if l.get("action") == "sent_email"}
                
                # Also keep track of draft IDs sent for safety
                sent_draft_ids = {l.get("draft_id") for l in logs if l.get("action") == "sent_email"}
                
                for entry in logs:
                    recipient = entry.get("recipient")
                    draft_id = entry.get("draft_id")
                    if entry.get("action") == "created_draft":
                        if recipient not in sent_recipients and draft_id not in sent_draft_ids:
                            # To prevent sending multiple drafts to the same recipient in this run,
                            # add to sent_recipients as we plan the send
                            drafts_to_send.append(entry)
                            sent_recipients.add(recipient)
        except Exception as e:
            print(f"Error reading activity log: {e}")
            return

    if not drafts_to_send:
        print("No pending drafts found in ACTIVITY_LOG.json to send.")
        return

    total = len(drafts_to_send)
    print(f"Verified {total} pending drafts. Starting send sequence following GUIDELINES.md...")
    
    for i, item in enumerate(drafts_to_send):
        draft_id = item['draft_id']
        recipient = item['recipient']
        
        try:
            print(f"[{i+1}/{total}] Sending draft {draft_id} to {recipient}...")
            # Use the drafts.send API
            sent_message = service.users().drafts().send(userId='me', body={'id': draft_id}).execute()
            
            message_id = sent_message.get('id')
            log_sent_activity(draft_id, message_id, recipient)
            print(f"Successfully sent! Message ID: {message_id}")
            
        except Exception as e:
            print(f"ERROR sending draft {draft_id}: {e}")
            # If a draft is no longer there (deleted or sent manually), log it or skip
            continue
            
        # VOLUME CONTROL: 1 per 30-60 seconds + 30-50% Jitter
        # Guidelines: "space them out (e.g., 1 message every 30-60 seconds) to mimic human behavior"
        # Guidelines: "Every delay... must include a 30-50% jitter."
        
        if i < total - 1: # Don't wait after the last one
            base_delay = random.uniform(30.0, 60.0)
            jitter_percent = random.uniform(0.3, 0.5)
            jitter = base_delay * jitter_percent
            
            # Add or subtract jitter
            total_delay = base_delay + (jitter if random.choice([True, False]) else -jitter)
            
            # Ensure we stay within reasonable bounds (min 20s, max 90s)
            total_delay = max(20.0, min(total_delay, 90.0))
            
            print(f"Waiting {total_delay:.2f} seconds before next send (Mimicry & Jitter)...")
            time.sleep(total_delay)

    print("\nMass outreach send sequence complete.")

if __name__ == "__main__":
    send_drafts()

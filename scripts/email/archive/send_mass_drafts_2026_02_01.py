import os
import time
import random
import json
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Keeping same Log File for continuity
LOG_FILE = "ACTIVITY_LOG.json"

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose', 
    'https://www.googleapis.com/auth/gmail.send' 
]

# The contacts we just created drafts for
TARGET_EMAILS = [
    "kmheyburn@gmail.com",
    "Bill@SOSAlliance.org",
    "parks-env@zilkerneighborhood.org",
    "chris@spyglassrealty.com",
    "chris@heartwoodre.com"
]

def log_sent_activity(draft_id, message_id, recipient):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "sent_email_retry",
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
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("Auth required.")
            return None
    return creds

def send_retry_drafts():
    creds = get_credentials()
    if not creds: return
    service = build('gmail', 'v1', credentials=creds)

    print("Searching for drafts to specific target emails...")
    
    # We will list drafts and check their 'To' header
    try:
        results = service.users().drafts().list(userId='me').execute()
        drafts = results.get('drafts', [])
    except Exception as e:
        print(f"Error listing drafts: {e}")
        return

    drafts_to_send = []
    
    for draft_meta in drafts:
        d_id = draft_meta['id']
        try:
            # Get draft details to check recipient
            draft_detail = service.users().drafts().get(userId='me', id=d_id).execute()
            message_payload = draft_detail['message']['payload']
            headers = message_payload['headers']
            
            to_header = next((h['value'] for h in headers if h['name'] == 'To'), None)
            
            if to_header:
                # Check if any of our target emails are in the To header
                matched_target = None
                for target in TARGET_EMAILS:
                    if target.lower() in to_header.lower():
                        matched_target = target
                        break
                
                if matched_target:
                    print(f"Found matching draft {d_id} for {to_header}")
                    drafts_to_send.append({
                        "id": d_id,
                        "recipient": to_header
                    })
                    
        except Exception as e:
            print(f"Error checking draft {d_id}: {e}")

    if not drafts_to_send:
        print("No matching drafts found.")
        return

    print(f"\nFound {len(drafts_to_send)} drafts to send. Proceeding...")
    
    for i, draft in enumerate(drafts_to_send):
        d_id = draft['id']
        recip = draft['recipient']
        
        try:
            print(f"Sending draft {d_id} to {recip}...")
            sent_msg = service.users().drafts().send(userId='me', body={'id': d_id}).execute()
            m_id = sent_msg['id']
            print(f"SENT! Message ID: {m_id}")
            log_sent_activity(d_id, m_id, recip)
            
        except Exception as e:
            print(f"FAILED to send draft {d_id}: {e}")

        # Small delay between sends
        if i < len(drafts_to_send) - 1:
            delay = random.uniform(5.0, 10.0)
            print(f"Waiting {delay:.1f}s...")
            time.sleep(delay)

    print("\nRetry send sequence complete.")

if __name__ == "__main__":
    send_retry_drafts()

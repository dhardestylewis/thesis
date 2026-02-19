import base64
import os
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("Credentials not valid.")
            return

    service = build('gmail', 'v1', credentials=creds)

    print("Searching for bounce messages...")
    # Search for typical bounce senders or subjects
    # query = 'from:mailer-daemon OR subject:"Delivery Status Notification" OR subject:"Address not found" OR subject:"Undeliverable"'
    # Narrowing down to relevant timeframe (since the mass send on Jan 23rd)
    query = 'after:2026/01/22 (from:mailer-daemon OR subject:"Delivery Status Notification" OR subject:"Undeliverable")'
    
    results = service.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])

    if not messages:
        print('No bounce messages found.')
        return

    print(f"Found {len(messages)} potential bounce messages. Analyzing content...")
    
    bounced_addresses = []

    for msg_meta in messages:
        msg = service.users().messages().get(userId='me', id=msg_meta['id'], format='full').execute()
        snippet = msg.get('snippet', '')
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(no subject)')
        print(f"\nScanning Bounce: {subject}")
        
        # Try to find the failed email in the snippet or body
        # Usually looks like "The email account that you tried to reach does not exist" or "Address not found"
        # We need to extract the email address causing the error.
        
        # Simple regex for email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        # Get body
        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data')
                    if data:
                        body += base64.urlsafe_b64decode(data).decode('utf-8')
        elif 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
            
        full_text = f"{subject} {snippet} {body}"
        
        # Find all emails in the body
        found_emails = re.findall(email_pattern, full_text)
        
        # Filter out my own emails and common system emails
        ignore_list = ['mailer-daemon@googlemail.com', 'dl3645@columbia.edu', 'danielhardestylewis@gmail.com', 'postmaster@']
        
        candidates = []
        for email in found_emails:
            if not any(ignored in email.lower() for ignored in ignore_list):
                candidates.append(email)
        
        # Usually the first candidate in a bounce message is the failed recipient, 
        # but let's just collect them all and see what matches our contact list later.
        if candidates:
            print(f"  -> Potential failed recipients: {candidates}")
            bounced_addresses.extend(candidates)
        else:
            print("  -> Could not extract specific email address from body.")

    # Remove duplicates
    bounced_addresses = list(set(bounced_addresses))
    
    print("\nSummary of detected bounced addresses:")
    for email in bounced_addresses:
        print(f"- {email}")
        
    # Save to file for the report generator to use
    with open("bounced_emails.json", "w") as f:
        import json
        json.dump(bounced_addresses, f)

if __name__ == '__main__':
    main()

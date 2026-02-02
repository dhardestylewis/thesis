import os
import base64
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    """Archives self-sent emails verbatim (.eml format)."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("Error: 'credentials.json' not found. Please follow the setup guide.")
                return
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)

    # Search for self-sent emails in the last month
    query = 'from:me to:(me OR dl3645@columbia.edu OR danielhardestylewis@gmail.com) after:2025/12/22'
    results = service.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])

    if not messages:
        print('No messages found.')
    else:
        print(f'Found {len(messages)} messages. Archiving...')
        if not os.path.exists('emails_verbatim'):
            os.makedirs('emails_verbatim')

        for msg in messages:
            # Fetch message in 'raw' format to get the verbatim RFC822 data
            raw_msg = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
            
            # Decode the base64url encoded raw message
            msg_bytes = base64.urlsafe_b64decode(raw_msg['raw'].encode('ASCII'))
            
            # Save as .eml file using the message ID as filename
            file_path = f"emails_verbatim/{msg['id']}.eml"
            with open(file_path, 'wb') as f:
                f.write(msg_bytes)
            print(f'Saved: {file_path}')

if __name__ == '__main__':
    main()

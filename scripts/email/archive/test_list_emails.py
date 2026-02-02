import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
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
            print("Credentials not valid or expired, and no refresh token available.")
            return

    service = build('gmail', 'v1', credentials=creds)

    print("Attempting to list 5 recent messages...")
    results = service.users().messages().list(userId='me', maxResults=5).execute()
    messages = results.get('messages', [])

    if not messages:
        print('No messages found.')
    else:
        print('Messages:')
        for msg in messages:
            # Get the headers to find the subject
            msg_detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            headers = msg_detail['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), '(no subject)')
            snippet = msg_detail.get('snippet', '')
            print(f'- {subject} | {snippet[:50]}...')

if __name__ == '__main__':
    main()

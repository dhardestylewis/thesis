import os
import base64
import time
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose'
]

# --- 1. DATA: Reader Shortlist from Reader_Outreach_Strategy.md ---
READERS = [
    {"name": "Michael J. Perles", "email": "mjp2225@columbia.edu", "type": "gsapp"},
    {"name": "Zachary Aarons", "email": "zda2102@columbia.edu", "type": "practitioner"},
    {"name": "Tomasz Piskorski", "email": "tp2252@gsb.columbia.edu", "type": "faculty"},
    {"name": "Christopher Mayer", "email": "cm310@gsb.columbia.edu", "type": "faculty"},
    {"name": "Stijn Van Nieuwerburgh", "email": "svnieuwe@gsb.columbia.edu", "type": "faculty"},
    {"name": "David Blei", "email": "david.blei@columbia.edu", "type": "faculty_technical"},
    {"name": "Zoran Kostic", "email": "zk2172@columbia.edu", "type": "faculty_technical"},
    {"name": "Emily Tolbert", "email": "elj2130@columbia.edu", "type": "gsapp"},
    {"name": "Matthew Bauer", "email": "mab2468@columbia.edu", "type": "gsapp"},
    {"name": "Alanna Browdy", "email": "aeb2217@columbia.edu", "type": "gsapp"},
    {"name": "Agostino Capponi", "email": "ac3827@columbia.edu", "type": "faculty_technical"}
]

def get_credentials():
    # Using the token from the workspace for convenience, if it exists
    token_path = r'c:\Users\dhl\.gemini\antigravity\scratch\gmail-mcp-workspace\token.json'
    creds_path = r'c:\Users\dhl\.gemini\antigravity\scratch\gmail-mcp-workspace\credentials.json'
    
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token_reader.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def create_drafts():
    creds = get_credentials()
    service = build('gmail', 'v1', credentials=creds)
    
    # Path to Overview PDF for attachment
    overview_path = r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Project_Overview.pdf"
    
    print(f"Creating drafts for {len(READERS)} potential readers...")

    for reader in READERS:
        full_name = reader['name']
        last_name = full_name.split(' ')[-1]
        email = reader['email']
        r_type = reader['type']
        
        subject = f"Thesis Reader Request: Predicting Zoning Opposition (Daniel Hardesty Lewis)"
        
        # Refined Unified Template
        body_html = f"""
        <p>Hi Professor {last_name},</p>
        
        <p>I am a Columbia M.S. Urban Planning student finishing a thesis that models and predicts zoning opposition (NIMBYism) at the owner- and parcel-level using machine learning and a 20-year database of letters in protest. I am reaching out to ask if you would consider serving as a thesis reader (+ $300 compensation).</p>
        
        <p>My goal is to produce a high-impact, defensible empirical workflow that (1) predicts opposition likelihood, (2) distinguishes business vs. individual owners, and (3) supports interpretable drivers of opposition. I would especially value your guidance on validity checks and the application of predictive analytics to political behavior.</p>
        
        <p>The commitment is lightweight: one brief meeting to align on scope and one read of the draft near submission. I've attached a project overview for your review.</p>
        
        <p>Would you be open to this?</p>
        
        <p>Best regards,</p>
        <p>Daniel Hardesty Lewis<br>
        Columbia University GSAPP<br>
        dl3645@columbia.edu</p>
        """

        message = MIMEMultipart()
        message['To'] = email
        message['Subject'] = subject
        message.attach(MIMEText(body_html, 'html'))
        
        # Attachment
        if os.path.exists(overview_path):
            with open(overview_path, 'rb') as f:
                filename = os.path.basename(overview_path)
                attachment = MIMEApplication(f.read(), _subtype='pdf')
                attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                message.attach(attachment)

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        draft_body = {'message': {'raw': raw_message}}
        
        try:
            draft = service.users().drafts().create(userId='me', body=draft_body).execute()
            print(f"✅ READER DRAFT CREATED: {full_name} ({email}) - ID: {draft['id']}")
        except Exception as e:
            print(f"❌ ERROR CREATING DRAFT for {full_name}: {e}")
            
        time.sleep(1)

if __name__ == "__main__":
    create_drafts()

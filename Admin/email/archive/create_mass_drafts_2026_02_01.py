import os
import base64
import time
import random
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose', 
    'https://www.googleapis.com/auth/gmail.send' 
]

# Updated Verified Contacts with "Fresher" Hooks
ALT_CONTACTS = [
    {
        "name": "Kevin Heyburn",
        "emails": ["kmheyburn@gmail.com"], 
        # Kept HPNA hook as it wasn't flagged as problematic, but smoothed it slightly.
        "hook": "Given HPNA's unique approach to <b>Neighborhood Conservation Combining Districts</b>, I am writing to request a brief interview to discuss how Hyde Park is navigating the tension between preservation and the city's recent push for density."
    },
    {
        "name": "Bill Bunch",
        "emails": ["Bill@SOSAlliance.org", "parks-env@zilkerneighborhood.org"],
        # Updated Hook: More general, referencing "various development projects" and recent environmental stands.
        "hook": "I've run into your past testimony on <b>various development projects and environmental ordinances</b> (including the recent West US 290 debate), and I am writing to request a brief interview to discuss the specific patterns of neighborhood and environmental concern you see effectively influencing Council decisions versus those that get ignored."
    },
    {
        "name": "Chris Affinito",
        "emails": ["chris@spyglassrealty.com", "chris@heartwoodre.com"],
        # Updated Hook: More general about his infill work (Springdale/Reyna) rather than one specific stale address.
        "hook": "I've run into your work on <b>various infill communities like The Reyna and Springdale</b>, and I am writing to request a brief interview to discuss the practical reality of permitting these projects and the specific neighborhood feedback that you find most challenging to address."
    }
]

def get_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def create_drafts():
    creds = get_credentials()
    service = build('gmail', 'v1', credentials=creds)
    
    attachment_paths = [
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Project_Overview.pdf",
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Interview_Guide.pdf"
    ]
    
    print(f"Creating REVISED drafts for {len(ALT_CONTACTS)} retry contacts...")

    for i, contact in enumerate(ALT_CONTACTS):
        name = contact['name']
        hook = contact['hook']
        emails = contact['emails']
        
        greeting = random.choice(["Dear", "Hi", "Hello"])
        closing = random.choice(["Best regards,", "Sincerely,", "Thank you,"])
        
        subject = "Interview request: Columbia research on Austin housing policy (Follow-up)"
        
        body_html = f"""
        <p>{greeting} {name},</p>
    
        <p>I am a graduate student at Columbia University predicting future housing opposition in Austin using AI.</p> 
    
        <p>{hook}</p>
        
        <p>Specifically, I am exploring how predictive tools and open data might influence (or interfere with) neighborhood engagement in zoning. Given your experience, I'd love to hear your thoughts.</p>
        
        <p>Would you be available for a <b>45‑60 minute</b> interview? This can be done via Zoom or in person during my Austin visit the <b>week of Feb 16–20</b>.</p>
        
        <p>I've attached a project overview and interview guide which explains the research objectives and exactly what topics we'll cover.</p>
        
        <p>Thank you for your time and consideration.</p>
        
        <p>{closing}</p>
        
        <p>Daniel Hardesty Lewis<br>
        Columbia University GSAPP<br>
        dl3645@columbia.edu</p>
        """

        for email in emails:
            message = MIMEMultipart()
            message['To'] = email
            message['Subject'] = subject
            message.attach(MIMEText(body_html, 'html'))
            
            for attachment_path in attachment_paths:
                with open(attachment_path, 'rb') as f:
                    filename = os.path.basename(attachment_path)
                    attachment = MIMEApplication(f.read(), _subtype='pdf')
                    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                    message.attach(attachment)

            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            draft_body = {'message': {'raw': raw_message}}
            
            try:
                draft = service.users().drafts().create(userId='me', body=draft_body).execute()
                print(f"REVISED DRAFT CREATED for {name} ({email}) - ID: {draft['id']}")
            except Exception as e:
                print(f"ERROR creating draft for {name} ({email}): {e}")
                
            time.sleep(2) 

if __name__ == "__main__":
    create_drafts()

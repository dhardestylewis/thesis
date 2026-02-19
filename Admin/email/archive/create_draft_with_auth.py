import os
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Scopes required for creating drafts
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose'
]

def get_credentials():
    """Gets valid user credentials from storage or triggers auth flow."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception:
            creds = None

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        
        if not creds:
            if not os.path.exists('credentials.json'):
                print("Error: 'credentials.json' missing.")
                return None
                
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return creds

def create_draft(creds):
    """Creates the draft email."""
    service = build('gmail', 'v1', credentials=creds)
    
    # Email content
    to_email = "dl3645@columbia.edu"
    subject = "[TEST] Thesis Research Participation Invitation - NIMBYism Study"
    
    body = """Dear Test Recipient,

I am a graduate student at Columbia University conducting research on neighborhood opposition to housing development in Austin, Texas. This study has been approved by Columbia University's Institutional Review Board (IRB Protocol #Y01M00).

I am reaching out to invite you to participate in a research interview about your experience with Austin's zoning and development processes. The interview would take approximately 45-60 minutes and can be conducted at a time and location convenient for you.

Your participation would contribute valuable insights to understanding the role of data and technology in urban planning decisions. All information will be kept confidential according to IRB protocols.

If you would be willing to participate, or if you have any questions, please feel free to contact me at dl3645@columbia.edu.

Thank you for considering this request.

Best regards,
Daniel Hardesty Lewis
Graduate Student, Urban Planning
Columbia University

IRB Contact: AskIRB@columbia.edu
"""
    
    # Create message
    message = MIMEMultipart()
    message['To'] = to_email
    message['Subject'] = subject
    
    # Add body
    message.attach(MIMEText(body, 'plain'))
    
    # Add attachment
    attachment_path = r"c:\Users\dhl\data\thesis\thesis\Submitted\IRB_Submitted\Protocol_and_Submission\Predicting_NIMBYism_Interview_Guide.pdf"
    
    if os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as f:
            attachment = MIMEApplication(f.read(), _subtype='pdf')
            attachment.add_header('Content-Disposition', 'attachment', 
                                filename='Predicting_NIMBYism_Interview_Guide.pdf')
            message.attach(attachment)
    else:
        print(f"Warning: Attachment not found at {attachment_path}")
    
    # Create draft
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    draft_body = {'message': {'raw': raw_message}}
    
    try:
        draft = service.users().drafts().create(userId='me', body=draft_body).execute()
        print(f"SUCCESS: Draft created with ID: {draft['id']}")
        return draft
    except Exception as e:
        print(f"Error creating draft: {e}")
        return None

if __name__ == "__main__":
    print("Starting Draft Creation...")
    creds = get_credentials()
    if creds:
        create_draft(creds)
    else:
        print("Failed to obtain credentials.")

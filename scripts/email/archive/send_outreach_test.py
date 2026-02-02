import os
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Scopes required for sending
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.send' 
]

def get_credentials():
    """Gets valid user credentials from storage or triggers auth flow."""
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
            if not os.path.exists('credentials.json'):
                print("Error: 'credentials.json' missing.")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return creds

def send_email():
    """Sends the test email with attachments."""
    creds = get_credentials()
    if not creds:
        return

    service = build('gmail', 'v1', credentials=creds)
    
    to_email = "dl3645@columbia.edu"
    subject = "Interview request: Columbia research on Austin housing policy"
    
    # HTML Body for formatting
    body_html = """
    <p>Dear Lauren Middleton‑Pratt,</p>

    <p>I am a graduate student at Columbia University, and I am predicting future housing opposition in Austin using AI.</p> 

    <p>Given your recent <b>Analysis of Density Bonus Programs</b> (Jan 2025), I am writing to request a brief interview regarding how city-led incentives interact with neighborhood-level opposition.</p>
    
    <p>Specifically, I am exploring how predictive tools and open data might influence (or interfere with) neighborhood engagement in zoning. Given your experience as Director of Planning, I'd love to hear your thoughts.</p>
    
    <p>Would you be available for a <b>45‑60 minute</b> interview? This can be done via Zoom or in person during my Austin visit the <b>week of Feb 16–20</b>.</p>
    
    <p>I've attached a project overview and interview guide which explains the research objectives and exactly what topics we'll cover.</p>
    
    <p>Thank you for your time and consideration.</p>
    
    <p>Best regards,</p>
    
    <p>Daniel Hardesty Lewis<br>
    Columbia University GSAPP<br>
    dl3645@columbia.edu</p>
    """
    
    message = MIMEMultipart()
    message['To'] = to_email
    message['Subject'] = subject
    
    message.attach(MIMEText(body_html, 'html'))
    
    # Attachments
    attachment_paths = [
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Project_Overview.pdf",
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Interview_Guide.pdf"
    ]
    
    for attachment_path in attachment_paths:
        if os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                filename = os.path.basename(attachment_path)
                attachment = MIMEApplication(f.read(), _subtype='pdf')
                attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                message.attach(attachment)
                print(f"Attached: {filename}")
        else:
            print(f"Error: Attachment not found at {attachment_path}")
            return

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    body = {'raw': raw_message}
    
    try:
        sent_message = service.users().messages().send(userId='me', body=body).execute()
        print(f"SUCCESS: Email sent! Message ID: {sent_message['id']}")
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == "__main__":
    send_email()

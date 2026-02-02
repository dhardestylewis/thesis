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
            
            # Note: If token.json exists but lacks 'send' scope, we might need to delete it manually or handle re-auth.
            # For this script, we assume if scopes mismatch we might need re-auth.
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return creds

def send_email():
    """Sends the test email with attachment."""
    creds = get_credentials()
    if not creds:
        return

    service = build('gmail', 'v1', credentials=creds)
    
    to_email = "dl3645@columbia.edu"
    subject = "[TEST SEND] Thesis Research Participation Invitation - NIMBYism Study"
    
    body = """Dear Daniel (Test),

This is a LIVE TEST of the thesis recruitment email functionality.

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
    
    message = MIMEMultipart()
    message['To'] = to_email
    message['Subject'] = subject
    
    message.attach(MIMEText(body, 'plain'))
    
    # Attachment: Interview Guide
    attachment_path = r"c:\Users\dhl\data\thesis\thesis\Submitted\IRB_Submitted\Protocol_and_Submission\Predicting_NIMBYism_Interview_Guide.pdf"
    
    if os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as f:
            attachment = MIMEApplication(f.read(), _subtype='pdf')
            attachment.add_header('Content-Disposition', 'attachment', 
                                filename='Predicting_NIMBYism_Interview_Guide.pdf')
            message.attach(attachment)
            print(f"Attached: {attachment_path}")
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
    # If token.json exists from previous draft-only scope, rename it to force re-auth with send scope
    # actually, 'compose' scope allows sending too, but adding 'send' scope to be explicit/safe.
    # To be safe, let's just run. If it fails with 403, we know why.
    # But better DX: Force re-auth if needed? 
    # Let's delete token.json if we are changing scopes? 
    # For now, I'll rely on the user instructions if it fails, or just include the logic.
    # Actually, let's delete token.json inside the script if we suspect issues? No, that's risky.
    # I will stick to the standard flow.
    send_email()

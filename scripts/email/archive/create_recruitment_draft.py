import os
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def create_draft_with_attachment():
    """Create a draft email with the recruitment template and Interview Guide attachment"""
    
    # Load credentials
    creds = Credentials.from_authorized_user_file('token.json', 
        ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.compose'])
    
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
    
    with open(attachment_path, 'rb') as f:
        attachment = MIMEApplication(f.read(), _subtype='pdf')
        attachment.add_header('Content-Disposition', 'attachment', 
                            filename='Predicting_NIMBYism_Interview_Guide.pdf')
        message.attach(attachment)
    
    # Create draft
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    draft_body = {'message': {'raw': raw_message}}
    
    draft = service.users().drafts().create(userId='me', body=draft_body).execute()
    
    print(f"Draft created successfully!")
    print(f"Draft ID: {draft['id']}")
    print(f"Message ID: {draft['message']['id']}")
    print(f"\nTo: {to_email}")
    print(f"Subject: {subject}")
    print(f"Attachment: Predicting_NIMBYism_Interview_Guide.pdf")

if __name__ == "__main__":
    create_draft_with_attachment()

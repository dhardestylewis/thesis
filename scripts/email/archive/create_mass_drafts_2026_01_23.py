import os
import base64
import time
import random
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Activity Log Path
LOG_FILE = "ACTIVITY_LOG.json"

# Scopes required for creating drafts
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose', 
    'https://www.googleapis.com/auth/gmail.send' 
]

# Contact Data extracted from Interview_Outreach_Guide.md
CONTACTS = [
    # City Officials & Planners
    {
        "name": "Lauren Middleton‑Pratt",
        "emails": ["lauren.middleton-pratt@austintexas.gov"],
        "hook": "Given your recent <b>Analysis of Density Bonus Programs</b> (Jan 2025), I am writing to request a brief interview regarding your department's experience managing the neighborhood feedback that arises during these rezoning processes."
    },
    {
        "name": "José Roig",
        "emails": ["jose.roig@austintexas.gov"],
        "hook": "Following your <b>HOME Initiative implementation report</b>, I am writing to request a brief interview to discuss the practical challenges of permitting under these new rules and the feedback you've received from the community."
    },
    {
        "name": "Joydeep Goswami",
        "emails": ["Joydeep.Goswami@austintexas.gov"],
        "hook": "Given your oversight of <b>affordable housing land development review</b>, I am writing to request a brief interview to discuss the specific sticking points or neighborhood concerns that typically surface during site plan reviews."
    },
    {
        "name": "Joi Harden",
        "emails": ["joi.harden@austintexas.gov"],
        "hook": "As the coordinator of <b>Planning Commission hearings</b>, I am writing to request a brief interview regarding the patterns of public testimony you've observed and the specific issues neighbors frequently raise."
    },
    {
        "name": "Jonathan Tomko",
        "emails": ["jonathan.tomko@austintexas.gov"],
        "hook": "Following your management of the <b>DB90 rezoning cases</b>, I am writing to request a brief interview to discuss the specific issues that made those projects particularly contested for local neighbors."
    },
    {
        "name": "Sherri Sirwaitis",
        "emails": ["sherri.sirwaitis@austintexas.gov"],
        "hook": "Given your experience managing <b>neighborhood consultation on zoning</b>, I am writing to request a brief interview to discuss the recurring concerns you hear from residents during these outreach sessions."
    },
    # Neighborhood Association Leaders
    {
        "name": "Steve Amos",
        "emails": ["oldwestaustin@gmail.com"],
        "hook": "Given OWANA’s <b>collaborative approach with developers</b>, I am writing to request a brief interview to discuss your steering committee's strategy for negotiating project changes that address neighborhood concerns."
    },
    {
        "name": "Kevin Heyburn",
        "emails": ["kevin.heyburn@austinhydepark.org", "kheyburn@austinhydepark.org", "president@austinhydepark.org"],
        "hook": "Given HPNA's work on <b>Neighborhood Conservation Combining Districts</b>, I am writing to request a brief interview to discuss Hyde Park's unique approach to handling density and the specific concerns your members have about current citywide reforms."
    },
    {
        "name": "Bill Bunch",
        "emails": ["bill.bunch@zilkerneighborhood.org", "bbunch@zilkerneighborhood.org", "info@zilkerneighborhood.org"],
        "hook": "Following your <b>testimony on the 2130 Goodrich Ave project</b> (Sept 2024), I am writing to request a brief interview to discuss the specific impacts that Zilker residents are worried about regarding these types of developments."
    },
    {
        "name": "Ana Aguirre",
        "emails": ["president@ancweb.org", "infoancatx@gmail.com"],
        "hook": "Given your work on the <b>HOME Initiative's impact on neighborhoods</b>, I am writing to request a brief interview to discuss the specific concerns ANC members have raised regarding their ability to influence the zoning process."
    },
    {
        "name": "Pat Whiteside",
        "emails": ["pat.whiteside@bartonhills.org", "pwhiteside@bartonhills.org", "bhna@bartonhills.org"],
        "hook": "Given your leadership during the <b>2025 zoning debates</b>, I am writing to request a brief interview to discuss how Barton Hills residents are organizing and the main issues they have with recent code changes."
    },
    # Housing Advocates (YIMBY)
    {
        "name": "Felicity Maxwell",
        "emails": ["fmaxwell@aura-atx.org", "felicity@texansforhousing.org", "info@aura-atx.org"],
        "hook": "Given your <b>role on the Planning Commission</b> and work with AURA, I am writing to request a brief interview to discuss the specific neighborhood pushback you've encountered and how those concerns play out in commission votes."
    },
    {
        "name": "Zach Faddis",
        "emails": ["zfaddis@aura-atx.org", "zach@aura-atx.org", "info@aura-atx.org"],
        "hook": "Following your <b>leadership on the HOME ordinance support</b>, I am writing to request a brief interview to discuss the main arguments you hear from opponents and the specific challenges of message-testing pro-housing policy."
    },
    {
        "name": "Sharad Mudhol",
        "emails": ["sharad.mudhol@gmail.com", "smudhol@gmail.com", "austininfillcoalition@gmail.com"],
        "hook": "Following your <b>testimony to the Planning Commission</b>, I am writing to request a brief interview to discuss the developer-side perspective on which neighborhood concerns create the most significant barriers for infill projects."
    },
    # Civic Technologists & Academics
    {
        "name": "Sherri Greenberg",
        "emails": ["srgreenberg@mail.utexas.edu"],
        "hook": "Given your work <b>chairing Good Systems</b>, I am writing to request a brief interview to discuss how people in Austin are reacting to the idea of using AI and predictive tools in city planning decisions."
    },
    {
        "name": "Junfeng Jiao",
        "emails": ["jjiao@austin.utexas.edu"],
        "hook": "Following your creation of <b>OpenCityAI</b>, I am writing to request a brief interview to discuss the feedback you've received from community members and city officials on the practical use of these data tools."
    },
    {
        "name": "Mateo Clarke",
        "emails": ["mateo@open-austin.org", "info@open-austin.org"],
        "hook": "Given your <b>OGP presentation on Austin's civic tech</b>, I am writing to request a brief interview to discuss how easy (or difficult) it really is for regular residents to use city data to engage with the planning process."
    },
    {
        "name": "Kerry O'Connor",
        "emails": ["kerry.oconnor@gmail.com", "koconnor@gmail.com", "kerry@kerryoconnor.com"],
        "hook": "Given your tenure as <b>Austin's Chief Innovation Officer</b>, I am writing to request a brief interview to discuss your experience trying to make city data more accessible and the hurdles you faced with public trust."
    },
    # Developers
    {
        "name": "Scott Turner",
        "emails": ["ASK@RIVERSIDEHOMESAUSTIN.COM", "sturner@riversidehomesaustin.com"],
        "hook": "Given your <b>20+ years of infill experience</b>, I am writing to request a brief interview to discuss the specific neighborhood concerns that most frequently stall or impact the feasibility of your projects."
    },
    {
        "name": "Chris Affinito",
        "emails": ["chris@heartwoodrealestate.co", "caffinito@heartwoodrealestate.co", "chris@heartwood.build"],
        "hook": "Following the <b>2130 Goodrich project</b>, I am writing to request a brief interview to discuss your experience working through the specific concerns raised by Zilker neighbors during that process."
    },
    {
        "name": "Taylor Smith",
        "emails": ["tsmith@abor.com"],
        "hook": "Following your <b>analysis of the HOME Initiative</b>, I am writing to request a brief interview to discuss what your members are hearing from their clients about neighborhood pushback and the impact on housing delivery."
    },
    {
        "name": "Mandy DeMayo",
        "emails": ["mandy.demayo@austintexas.gov", "housing@austintexas.gov"],
        "hook": "Given your oversight of the <b>Equitable Development Initiative</b>, I am writing to request a brief interview to discuss how your team handles community concerns about gentrification and neighborhood change in newly upzoned areas."
    },
    {
        "name": "Foundation Communities Representative",
        "emails": ["info@foundcom.org", "outreach@foundcom.org"],
        "hook": "With reference to the <b>Lamar Square project</b>, I am writing to request a brief interview to discuss how you've been working with the neighbors and the main concerns they've brought to the table."
    },
    {
        "name": "Austin Habitat for Humanity",
        "emails": ["info@austinhabitat.org", "outreach@austinhabitat.org"],
        "hook": "Given your <b>partnership with the City</b> on ownership housing, I am writing to request a brief interview to discuss the specific neighborhood pushback you face as a nonprofit developer compared to the private market."
    }
]

def log_activity(draft_id, recipient, action="created_draft"):
    """Logs the activity to JSON file as per guidelines."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "draft_id": draft_id,
        "recipient": recipient
    }
    
    data = []
    is_dict = False
    
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                content = json.load(f)
                if isinstance(content, dict):
                    data = content.get("activities", [])
                    is_dict = True
                elif isinstance(content, list):
                    data = content
        except json.JSONDecodeError:
            pass
            
    data.append(entry)
    
    if is_dict:
        with open(LOG_FILE, 'r') as f:
            content = json.load(f)
        content["activities"] = data
        with open(LOG_FILE, 'w') as f:
            json.dump(content, f, indent=2)
    else:
        with open(LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)

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

def create_drafts():
    """Creates the outreach drafts."""
    creds = get_credentials()
    if not creds:
        return

    service = build('gmail', 'v1', credentials=creds)
    
    # Attachments paths
    attachment_paths = [
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Project_Overview.pdf",
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Interview_Guide.pdf"
    ]
    
    total_contacts = len(CONTACTS)
    print(f"Starting draft creation for {total_contacts} contacts with randomized delays (jitter)...")
    
    # Load existing log to skip duplicates
    existing_emails = set()
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                content = json.load(f)
                logs = content.get("activities", []) if isinstance(content, dict) else content
                for entry in logs:
                    if entry.get("action") == "created_draft":
                        existing_emails.add(entry.get("recipient"))
        except Exception:
            pass

    for i, contact in enumerate(CONTACTS):
        name = contact['name']
        hook = contact['hook']
        emails = contact['emails']
        
        # Metadata Hygiene: Vary greetings and closings slightly
        greeting = random.choice(["Dear", "Hi", "Hello"])
        closing = random.choice(["Best regards,", "Sincerely,", "Thank you,"])
        
        subject = "Interview request: Columbia research on Austin housing policy"
        
        # HTML Body
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

        # Verify attachments exist before loop
        for path in attachment_paths:
            if not os.path.exists(path):
                print(f"CRITICAL ERROR: Attachment not found at {path}")
                return

        # Create a draft for EACH email address in the list
        for email in emails:
            if email in existing_emails:
                print(f"[{i+1}/{total_contacts}] SKIPPING {name} ({email}) - Draft already exists in log.")
                continue
            message = MIMEMultipart()
            message['To'] = email
            message['Subject'] = subject
            
            message.attach(MIMEText(body_html, 'html'))
            
            # Add Attachments
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
                print(f"[{i+1}/{total_contacts}] DRAFT CREATED for {name} ({email}) - ID: {draft['id']}")
                log_activity(draft['id'], email)
            except Exception as e:
                print(f"ERROR creating draft for {name} ({email}): {e}")
                
            # Random "Jitter" Delay: 5 to 12 seconds between drafts
            # This mimics "Contextual Thinking Time" for creating new drafts
            delay = random.uniform(5.0, 12.0)
            time.sleep(delay)

    print("\nAll drafts created successfully. Check 'ACTIVITY_LOG.json' for records.")

if __name__ == "__main__":
    create_drafts()

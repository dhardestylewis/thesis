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

# --- 1. DATA: Non-Responders with Alternate Routes ---
# Each entry has 'name', primary 'email', optional 'cc', 'route_type' (direct, assistant, sms), and 'hook'.

TARGETS = [
    # === City Officials (CC Strategy) ===
    {
        "name": "Lauren Middleton-Pratt",
        "email": "lauren.middleton-pratt@austintexas.gov",
        "cc": ["christopher.crain@austintexas.gov", "erica.leak@austintexas.gov"], 
        "route": "cc_colleague",
        "hook": "As Director, your perspective on how the Planning Department balances long-range comprehensive planning with immediate zoning case pressures is critical for my model.",
        "role": "Director of Planning"
    },
    {
        "name": "Jose Roig",
        "email": "jose.roig@austintexas.gov",
        "route": "direct", # No specific CC found, direct is best
        "hook": "As DSD Director, your insight into the friction points between code requirements and development timelines is exactly what my research aims to quantify.",
        "role": "Director of DSD"
    },
    {
        "name": "Sherri Sirwaitis",
        "email": "sherri.sirwaitis@austintexas.gov",
        "cc": ["christopher.crain@austintexas.gov"],
        "route": "cc_colleague",
        "hook": "Your work on current planning cases places you at the intersection of policy and practice.",
        "role": "Planner Principal"
    },
    
    # === City Officials (Assistant/Deputy Strategy) ===
    # For Goswami/Harden, we are cross-referencing them.
    {
        "name": "Joydeep Goswami",
        "email": "Joydeep.Goswami@austintexas.gov",
        "cc": ["joi.harden@austintexas.gov"],
        "route": "cc_colleague",
        "hook": "Your operational oversight at DSD is vital to understanding the 'ground truth' of housing production.",
        "role": "DSD Executive"
    },
    {
        "name": "Joi Harden",
        "email": "joi.harden@austintexas.gov",
        "cc": ["Joydeep.Goswami@austintexas.gov"],
        "route": "cc_colleague",
        "hook": "Your role as Executive Liaison informs the critical bridge between council policy and departmental execution.",
        "role": "Executive Liaison"
    },

    # === Neighborhood Leaders (Alternate Email / Colleague) ===
    {
        "name": "Steve Amos",
        "email": "oldwestaustin@gmail.com",
        "cc": [], # Addressing Renae in body is better logic for generic email
        "route": "direct_context", 
        "context_note": "ATTN: Renae Alsobrook (Membership Chair) - please forward to Steve.",
        "hook": "As OWANA Parks Chair, your views on how greenspace preservation interacts with infill pressure are unique.",
        "role": "OWANA Parks Chair"
    },
    {
        "name": "Ana Aguirre",
        "email": "president@atxanc.org", # NEW EMAIL
        "cc": ["admin@atxanc.org"],
        "route": "direct",
        "hook": "As President of the ANC, you have the broadest view of how neighborhood feedback mechanisms are evolving citywide.",
        "role": "ANC President"
    },
    {
        "name": "Pat Whiteside",
        "email": "officers@bartonhills.org", # NEW EMAIL
        "route": "direct_context",
        "context_note": "ATTN: Barb LaFollette (President) - please ensure Pat sees this.",
        "hook": "Barton Hills has been a focal point for environmental vs density debates.",
        "role": "BHNA Vice-President"
    },

    # === Advocates & Developers (Alternate Emails) ===
    {
        "name": "Sharad Mudhol",
        "email": "info@smartDigsAustin.com", # NEW EMAIL
        "route": "direct",
        "hook": "Smart Digs' experience with infill development provides a perfect case study for the friction costs I am modeling.",
        "role": "Smart Digs"
    },
    {
        "name": "Scott Turner",
        "email": "scott@turnerresidential.com", # NEW EMAIL
        "route": "direct",
        "hook": "Your leadership at Turner Residential and Riverside Homes puts you at the forefront of the practical challenges in delivering housing.",
        "role": "Turner Residential"
    },
    {
        "name": "Foundation Communities",
        "email": "Norris.Deajon@foundcom.org", # NEW EMAIL (Comms Manager)
        "route": "assistant_request", # Requesting interview with leadership
        "hook": "Foundation Communities is the gold standard for affordable housing in Austin.",
        "target_boss": "Executive Team",
        "role": "Communications Manager"
    },
    {
        "name": "Austin Habitat",
        "email": "aleverett@austinhabitat.org", # NEW EMAIL (VP Marketing)
        "route": "assistant_request",
        "hook": "Habitat's model of homeownership faces unique zoning challenges that I want to understand.",
        "target_boss": "Kelly Weiss (CEO)",
        "role": "VP Marketing"
    },
    
    # === Academics & Civil Servants (New/Official Emails) ===
    {
        "name": "Kerry O'Connor",
        "email": "kerry.oconnor@austintexas.gov", # NEW OFFICIAL EMAIL
        "route": "direct",
        "hook": "As Chief Innovation Officer, your work on open government and 'innovation' is the exact mechanism I am studying.",
        "role": "Chief Innovation Officer"
    },
    {
        "name": "Junfeng Jiao",
        "email": "saleh.afroogh@utexas.edu", # Lab Contact
        "route": "assistant_request",
        "target_boss": "Dr. Jiao",
        "hook": "I am a researcher at Columbia GSAPP using AI for housing prediction and would love to compare notes with the Urban Info Lab.",
        "role": "Lab Contact"
    },
    {
        "name": "Sherri Greenberg",
        "email": "srgreenberg@austin.utexas.edu", # Alternate UT email
        "route": "direct",
        "hook": "Your work at the LBJ School on state-local policy interplay is foundational to my research.",
        "role": "Professor"
    },
    
    # === EXPERIMENTAL: SMS Gateways ===
    {
        "name": "Zach Faddis (SMS)",
        "email": "2102641093@txt.att.net", # Trying AT&T first
        "route": "sms_short",
        "hook": "Daniel from Columbia Univ here. Hoping to chat about AURA and housing data.",
        "role": "AURA President"
    },
    {
        "name": "Jose Roig (SMS)",
        "email": "5124228451@txt.att.net", # Trying AT&T first
        "route": "sms_short",
        "hook": "Daniel from Columbia Univ here. Researching Dev Services friction/timelines.",
        "role": "DSD Director"
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

def check_duplicate(service, email):
    """
    Check if we have already sent an email to this address or have a draft for them.
    Returns True if duplicate found, False otherwise.
    """
    # 1. Check Sent Messages (last 30 days roughly to be safe)
    query_sent = f"to:{email} after:2026/01/01"
    try:
        results = service.users().messages().list(userId='me', q=query_sent).execute()
        if results.get('resultSizeEstimate', 0) > 0:
            print(f"⚠️  DUPLICATE DETECTED: Found sent message to {email}. Skipping.")
            return True
    except Exception as e:
        print(f"Error checking sent messages for {email}: {e}")

    # 2. Check Drafts
    # Drafts listing is harder to filter by 'to', so we list all and check headers (expensive but safe)
    # Validating duplicates in drafts is a bit slower but safer.
    # Note: For massive lists, this is inefficient. For 17 people, it's fine.
    try:
        drafts = service.users().drafts().list(userId='me').execute()
        if 'drafts' in drafts:
            for d in drafts['drafts']:
                d_detail = service.users().drafts().get(userId='me', id=d['id']).execute()
                headers = d_detail['message']['payload']['headers']
                for h in headers:
                    if h['name'] == 'To' and email in h['value']:
                         print(f"⚠️  DUPLICATE DETECTED: Found existing draft to {email}. Skipping.")
                         return True
    except Exception as e:
        print(f"Error checking drafts for {email}: {e}")
        
    return False

def create_drafts():
    creds = get_credentials()
    service = build('gmail', 'v1', credentials=creds)
    
    attachment_paths = [
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Project_Overview.pdf",
        r"c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Project_One_Pager\Lewis_Thesis_Interview_Guide.pdf"
    ]
    
    print(f"Creating REVISED drafts for {len(TARGETS)} non-responder alternates...")

    for i, contact in enumerate(TARGETS):
        name = contact['name']
        email = contact['email']
        route = contact['route']
        hook = contact['hook']
        role = contact.get('role', 'Stakeholder')
        
        # --- PRE-FLIGHT CHECK ---
        if check_duplicate(service, email):
            continue
            
        greeting = random.choice(["Dear", "Hi", "Hello"])
        closing = random.choice(["Best regards,", "Sincerely,", "Thank you,"])
        
        # Construct Subject & Body based on Route
        cc_list = contact.get('cc', [])
        
        if route == 'sms_short':
            subject = "Research Inquiry: Austin Housing Data"
            body_html = f"{hook} - Daniel Lewis (dl3645@columbia.edu)" # Plain text for SMS effectively
        
        elif route == 'assistant_request':
            target_boss = contact.get('target_boss', 'your Director')
            subject = f"Interview Scheduling for {target_boss}: Columbia Research"
            body_html = f"""
            <p>{greeting} {name},</p>
            <p>I am a graduate student at Columbia University hoping to schedule a brief interview with <b>{target_boss}</b> regarding Austin's housing policy and data.</p>
            <p>{hook}</p>
            <p>Would it be possible to find 30-45 minutes on their calendar during the <b>week of Feb 16–20</b>? I will be in Austin and can meet in person or via Zoom.</p>
            <p>I have attached a project overview for their review.</p>
            <p>{closing}</p>
            <p>Daniel Hardesty Lewis<br>Columbia University GSAPP</p>
            """

        else: # Direct, CC, or Context
            subject = "Interview request: Columbia research on Austin housing policy (Follow-up)"
            
            context_intro = ""
            if route == 'cc_colleague':
                cc_names = ", ".join([email.split('@')[0].replace('.', ' ').title() for email in cc_list])
                context_intro = f"<p><i>(I have copied {cc_names} on this note as I understand they work closely with you on these matters.)</i></p>"
            elif route == 'direct_context':
                context_intro = f"<p><i>{contact.get('context_note', '')}</i></p>"

            body_html = f"""
            <p>{greeting} {name},</p>
            {context_intro}
            <p>I am a graduate student at Columbia University predicting future housing opposition in Austin using AI.</p> 
            <p>{hook}</p>
            <p>Specifically, I am exploring how predictive tools and open data might influence (or interfere with) neighborhood engagement in zoning. Given your role as {role}, I'd love to hear your thoughts.</p>
            <p>Would you be available for a <b>45‑60 minute</b> interview? This can be done via Zoom or in person during my Austin visit the <b>week of Feb 16–20</b>.</p>
            <p>I've attached a project overview and interview guide which explains the research objectives and exactly what topics we'll cover.</p>
            <p>Thank you for your time.</p>
            <p>{closing}</p>
            <p>Daniel Hardesty Lewis<br>
            Columbia University GSAPP<br>
            dl3645@columbia.edu</p>
            """

        # Create Message
        message = MIMEMultipart()
        message['To'] = email
        if cc_list:
            message['Cc'] = ", ".join(cc_list)
        message['Subject'] = subject
        message.attach(MIMEText(body_html, 'html'))
        
        # Attachments (Skip for SMS)
        if route != 'sms_short':
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
            print(f"✅ DRAFT CREATED for {name} ({email}) [Route: {route}] - ID: {draft['id']}")
        except Exception as e:
            print(f"❌ ERROR creating draft for {name} ({email}): {e}")
            
        time.sleep(2) 

if __name__ == "__main__":
    create_drafts()

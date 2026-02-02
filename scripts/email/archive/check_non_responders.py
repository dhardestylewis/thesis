import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Full contact list from Jan 23
CONTACTS = [
    # City Officials & Planners
    {"name": "Lauren Middleton‑Pratt", "emails": ["lauren.middleton-pratt@austintexas.gov"]},
    {"name": "José Roig", "emails": ["jose.roig@austintexas.gov"]},
    {"name": "Joydeep Goswami", "emails": ["Joydeep.Goswami@austintexas.gov"]},
    {"name": "Joi Harden", "emails": ["joi.harden@austintexas.gov"]},
    {"name": "Jonathan Tomko", "emails": ["jonathan.tomko@austintexas.gov"]},
    {"name": "Sherri Sirwaitis", "emails": ["sherri.sirwaitis@austintexas.gov"]},
    # Neighborhood Association Leaders
    {"name": "Steve Amos", "emails": ["oldwestaustin@gmail.com"]},
    {"name": "Kevin Heyburn", "emails": ["kevin.heyburn@austinhydepark.org", "kheyburn@austinhydepark.org", "president@austinhydepark.org"]},
    {"name": "Bill Bunch", "emails": ["bill.bunch@zilkerneighborhood.org", "bbunch@zilkerneighborhood.org", "info@zilkerneighborhood.org"]},
    {"name": "Ana Aguirre", "emails": ["president@ancweb.org", "infoancatx@gmail.com"]},
    {"name": "Pat Whiteside", "emails": ["pat.whiteside@bartonhills.org", "pwhiteside@bartonhills.org", "bhna@bartonhills.org"]},
    # Housing Advocates (YIMBY)
    {"name": "Felicity Maxwell", "emails": ["fmaxwell@aura-atx.org", "felicity@texansforhousing.org", "info@aura-atx.org"]},
    {"name": "Zach Faddis", "emails": ["zfaddis@aura-atx.org", "zach@aura-atx.org", "info@aura-atx.org"]},
    {"name": "Sharad Mudhol", "emails": ["sharad.mudhol@gmail.com", "smudhol@gmail.com", "austininfillcoalition@gmail.com"]},
    # Civic Technologists & Academics
    {"name": "Sherri Greenberg", "emails": ["srgreenberg@mail.utexas.edu"]},
    {"name": "Junfeng Jiao", "emails": ["jjiao@austin.utexas.edu"]},
    {"name": "Mateo Clarke", "emails": ["mateo@open-austin.org", "info@open-austin.org"]},
    {"name": "Kerry O'Connor", "emails": ["kerry.oconnor@gmail.com", "koconnor@gmail.com", "kerry@kerryoconnor.com"]},
    # Developers
    {"name": "Scott Turner", "emails": ["ASK@RIVERSIDEHOMESAUSTIN.COM", "sturner@riversidehomesaustin.com"]},
    {"name": "Chris Affinito", "emails": ["chris@heartwoodrealestate.co", "caffinito@heartwoodrealestate.co", "chris@heartwood.build"]},
    {"name": "Taylor Smith", "emails": ["tsmith@abor.com"]},
    {"name": "Mandy DeMayo", "emails": ["mandy.demayo@austintexas.gov", "housing@austintexas.gov"]},
    {"name": "Foundation Communities Representative", "emails": ["info@foundcom.org", "outreach@foundcom.org"]},
    {"name": "Austin Habitat for Humanity", "emails": ["info@austinhabitat.org", "outreach@austinhabitat.org"]},
]

# Exclude contacts we JUST reached out to today (Feb 1)
EXCLUDE_NAMES = ["Kevin Heyburn", "Bill Bunch", "Chris Affinito"]

def main():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("Credentials invalid.")
            return

    service = build('gmail', 'v1', credentials=creds)

    print("Checking for responses from original batch (since Jan 23)...")
    
    non_responders = []
    
    for contact in CONTACTS:
        name = contact['name']
        
        if name in EXCLUDE_NAMES:
            continue
            
        emails = contact['emails']
        
        # Check if we received ANY email from ANY of their addresses after Jan 22
        has_responded = False
        
        for email in emails:
            query = f"from:{email} after:2026/01/22"
            try:
                results = service.users().messages().list(userId='me', q=query).execute()
                if results.get('resultSizeEstimate', 0) > 0:
                    has_responded = True
                    print(f"✅ Response found from {name} ({email})")
                    break # One response is enough to count as managed
            except Exception as e:
                print(f"Error checking {email}: {e}")

        if not has_responded:
            non_responders.append(contact)

    # Output Markdown Report
    with open("non_responders_report.md", "w", encoding="utf-8") as f:
        f.write("# Non-Responders Report\n\n")
        f.write("The following contacts were emailed on Jan 23rd but have **not responded** (and were not part of today's re-outreach).\n\n")
        f.write("| Name | Emails |\n")
        f.write("|---|---|\n")
        
        for p in non_responders:
            emails_str = "<br>".join(p['emails'])
            f.write(f"| {p['name']} | {emails_str} |\n")
    
    print(f"\nFound {len(non_responders)} non-responders. Report saved to 'non_responders_report.md'.")

if __name__ == "__main__":
    main()

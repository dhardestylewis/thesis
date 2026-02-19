import json
import os

# Define the contacts list directly from the source file content I just read
CONTACTS = [
    # City Officials & Planners
    {
        "name": "Lauren Middleton‑Pratt",
        "emails": ["lauren.middleton-pratt@austintexas.gov"],
    },
    {
        "name": "José Roig",
        "emails": ["jose.roig@austintexas.gov"],
    },
    {
        "name": "Joydeep Goswami",
        "emails": ["Joydeep.Goswami@austintexas.gov"],
    },
    {
        "name": "Joi Harden",
        "emails": ["joi.harden@austintexas.gov"],
    },
    {
        "name": "Jonathan Tomko",
        "emails": ["jonathan.tomko@austintexas.gov"],
    },
    {
        "name": "Sherri Sirwaitis",
        "emails": ["sherri.sirwaitis@austintexas.gov"],
    },
    # Neighborhood Association Leaders
    {
        "name": "Steve Amos",
        "emails": ["oldwestaustin@gmail.com"],
    },
    {
        "name": "Kevin Heyburn",
        "emails": ["kevin.heyburn@austinhydepark.org", "kheyburn@austinhydepark.org", "president@austinhydepark.org"],
    },
    {
        "name": "Bill Bunch",
        "emails": ["bill.bunch@zilkerneighborhood.org", "bbunch@zilkerneighborhood.org", "info@zilkerneighborhood.org"],
    },
    {
        "name": "Ana Aguirre",
        "emails": ["president@ancweb.org", "infoancatx@gmail.com"],
    },
    {
        "name": "Pat Whiteside",
        "emails": ["pat.whiteside@bartonhills.org", "pwhiteside@bartonhills.org", "bhna@bartonhills.org"],
    },
    # Housing Advocates (YIMBY)
    {
        "name": "Felicity Maxwell",
        "emails": ["fmaxwell@aura-atx.org", "felicity@texansforhousing.org", "info@aura-atx.org"],
    },
    {
        "name": "Zach Faddis",
        "emails": ["zfaddis@aura-atx.org", "zach@aura-atx.org", "info@aura-atx.org"],
    },
    {
        "name": "Sharad Mudhol",
        "emails": ["sharad.mudhol@gmail.com", "smudhol@gmail.com", "austininfillcoalition@gmail.com"],
    },
    # Civic Technologists & Academics
    {
        "name": "Sherri Greenberg",
        "emails": ["srgreenberg@mail.utexas.edu"],
    },
    {
        "name": "Junfeng Jiao",
        "emails": ["jjiao@austin.utexas.edu"],
    },
    {
        "name": "Mateo Clarke",
        "emails": ["mateo@open-austin.org", "info@open-austin.org"],
    },
    {
        "name": "Kerry O'Connor",
        "emails": ["kerry.oconnor@gmail.com", "koconnor@gmail.com", "kerry@kerryoconnor.com"],
    },
    # Developers
    {
        "name": "Scott Turner",
        "emails": ["ASK@RIVERSIDEHOMESAUSTIN.COM", "sturner@riversidehomesaustin.com"],
    },
    {
        "name": "Chris Affinito",
        "emails": ["chris@heartwoodrealestate.co", "caffinito@heartwoodrealestate.co", "chris@heartwood.build"],
    },
    {
        "name": "Taylor Smith",
        "emails": ["tsmith@abor.com"],
    },
    {
        "name": "Mandy DeMayo",
        "emails": ["mandy.demayo@austintexas.gov", "housing@austintexas.gov"],
    },
    {
        "name": "Foundation Communities Representative",
        "emails": ["info@foundcom.org", "outreach@foundcom.org"],
    },
    {
        "name": "Austin Habitat for Humanity",
        "emails": ["info@austinhabitat.org", "outreach@austinhabitat.org"],
    }
]

LOG_FILE = "ACTIVITY_LOG.json"

def main():
    if not os.path.exists(LOG_FILE):
        print("Error: ACTIVITY_LOG.json not found.")
        return

    with open(LOG_FILE, 'r') as f:
        try:
            log_data = json.load(f)
            activities = log_data.get("activities", []) if isinstance(log_data, dict) else log_data
        except json.JSONDecodeError:
            activities = []

    # Map Recipient -> Status
    # Priorities: SENT > DRAFT > NONE
    recipient_status = {}
    
    # First pass: Check for Drafts
    for entry in activities:
        if entry.get("action") == "created_draft":
            rec = entry.get("recipient")
            if rec:
                recipient_status[rec] = "Draft Created"

    # Third pass: Check for Bounces (overwrite Sent)
    bounced_list = []
    if os.path.exists("bounced_emails.json"):
        with open("bounced_emails.json", "r") as f:
            try:
                bounced_list = json.load(f)
                # Normalize to lower case for comparison
                bounced_list = [e.lower() for e in bounced_list]
            except:
                pass

    for email in recipient_status.keys():
        if email.lower() in bounced_list:
            recipient_status[email] = "BOUNCED (Returned to Sender)"

    # Build the Table Rows
    rows = []
    failed_rows = []

    for contact in CONTACTS:
        name = contact['name']
        for email in contact['emails']:
            # Check status
            status = recipient_status.get(email)
            if not status:
                # Try case insensitive match against keys
                for key in recipient_status:
                    if key.lower() == email.lower():
                        status = recipient_status[key]
                        break
            
            # Check bounce again loosely just in case log didn't capture it accurately but bounce file did
            if email.lower() in bounced_list:
                status = "BOUNCED (Returned to Sender)"

            final_status = status if status else "Not Attempted"
            
            rows.append(f"| {name} | {email} | {final_status} |")
            
            if final_status != "Sent":
                failed_rows.append(f"| {name} | {email} | {final_status} |")

    # Output Markdown to file
    with open("report.md", "w", encoding="utf-8") as f:
        f.write("# Interview Request Status Report\n\n")
        f.write("## ❌ Failed / Bounced Contacts\n")
        if failed_rows:
            f.write("| Name | Email | Status |\n")
            f.write("|---|---|---|\n")
            for row in failed_rows:
                f.write(row + "\n")
        else:
            f.write("No failures found. All contacts were sent successfully.\n")

        f.write("\n## 📋 All Attempted Emails\n")
        f.write("| Name | Email | Status |\n")
        f.write("|---|---|---|\n")
        for row in rows:
            f.write(row + "\n")
    
    print("Report generated: report.md")

if __name__ == "__main__":
    main()

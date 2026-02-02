import json
import os

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

def main():
    bounced_list = []
    if os.path.exists("bounced_emails.json"):
        with open("bounced_emails.json", "r") as f:
            try:
                bounced_list = json.load(f)
                bounced_list = [e.lower() for e in bounced_list]
            except:
                pass

    # Output Markdown to file
    with open("reachability_report.md", "w", encoding="utf-8") as f:
        f.write("# Contact Reachability Report\n\n")
        f.write("| Name | Reachable? | Successful Emails | Bounced Emails |\n")
        f.write("|---|---|---|---|\n")

        overall_stats = {"fully_reached": 0, "partially_reached": 0, "unreachable": 0}

        for contact in CONTACTS:
            name = contact['name']
            emails = contact['emails']
            
            bounced_count = 0
            successful_emails = []
            failed_emails = []
            
            for email in emails:
                if email.lower() in bounced_list:
                    bounced_count += 1
                    failed_emails.append(email)
                else:
                    successful_emails.append(email)
            
            total = len(emails)
            
            if bounced_count == total:
                status = "🔴 UNREACHABLE (All failed)"
                overall_stats["unreachable"] += 1
            elif bounced_count > 0:
                status = "🟡 PARTIAL (Some bounced)"
                overall_stats["partially_reached"] += 1
            else:
                status = "🟢 REACHED"
                overall_stats["fully_reached"] += 1
                
            success_str = "<br>".join(successful_emails) if successful_emails else "None"
            fail_str = "<br>".join(failed_emails) if failed_emails else "None"
            
            f.write(f"| {name} | {status} | {success_str} | {fail_str} |\n")

        f.write("\n## Summary\n")
        f.write(f"- Fully Reachable: {overall_stats['fully_reached']}\n")
        f.write(f"- Partially Reachable: {overall_stats['partially_reached']}\n")
        f.write(f"- Completely Unreachable: {overall_stats['unreachable']}\n")
        
        f.write("\n## Details for Unreachable Contacts\n")
        for contact in CONTACTS:
            name = contact['name']
            emails = contact['emails']
            bounced_count = sum(1 for e in emails if e.lower() in bounced_list)
            if bounced_count == len(emails):
                f.write(f"**{name}**\n")
                f.write("- All attempts failed:\n")
                for e in emails:
                    f.write(f"  - {e}\n")
                f.write("\n")
    
    print("Report generated: reachability_report.md")

if __name__ == "__main__":
    main()

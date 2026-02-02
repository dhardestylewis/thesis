import json
import os
from collections import Counter

LOG_FILE = "ACTIVITY_LOG.json"

def audit():
    if not os.path.exists(LOG_FILE):
        print("Log file not found.")
        return

    with open(LOG_FILE, 'r') as f:
        content = json.load(f)
        activities = content.get("activities", []) if isinstance(content, dict) else content

    sends = [a for a in activities if a.get("action") == "sent_email"]
    recipients = [s.get("recipient") for s in sends]
    counts = Counter(recipients)

    duplicates = {k: v for k, v in counts.items() if v > 1}

    print(f"Total Unique Recipients Sent To: {len(counts)}")
    print(f"Total Emails Sent: {len(sends)}")
    print("\nRECIPIENTS WITH DUPLICATE SENDS:")
    print("-" * 50)
    for email, count in duplicates.items():
        print(f"{email}: {count} sends")
    
    if not duplicates:
        print("No duplicates found.")

if __name__ == "__main__":
    audit()

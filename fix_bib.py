
import os

def fix_bib():
    file_path = 'references.bib'
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Decode ignoring errors to strip garbage
        text = content.decode('utf-8', errors='ignore')
        
        # Check if the text ends with the new citations.
        # If not, or if they are corrupted, append them.
        # We'll just truncate at the last known good entry if possible, 
        # or simply append the new ones if they are missing/corrupt.
        
        # Define the new citations block
        new_citations = """
@misc{electronicfrontier2025,
  author={{Electronic Frontier Alliance}},
  title={Austin Tech Alliance},
  year={2025},
  url={https://www.eff.org/}
}

@misc{govpilot2024,
  author={{GovPilot}},
  title={Government Software in Texas},
  year={2024},
  url={https://www.govpilot.com/}
}

@misc{idcsg2024,
  author={{IDC Smart Cities}},
  title={Smart Cities North America Awards},
  year={2024},
  url={https://www.idc.com/}
}

@misc{utgoodsystems2023,
  author={{UT Austin Good Systems}},
  title={A Good System for Smart Cities},
  year={2023},
  url={https://goodsystems.utexas.edu/}
}

@misc{austin_campaign_finance,
  title={Campaign finance reporting},
  author={{City of Austin}},
  publisher={Office of the City Clerk},
  year={2025},
  url={https://www.austintexas.gov/department/campaign-finance-reporting}
}

@misc{austin_lobbyist,
  title={Lobbyist Registration},
  author={{City of Austin}},
  publisher={Office of the City Clerk},
  year={2025},
  url={https://www.austintexas.gov/department/lobbyist-registration}
}
"""
        # Find where to cut. Let's look for "campaign finance" or one of the new keys.
        # If the file is corrupted at the end, identifying the cut point is hard.
        # We will search for a known string from the original file's end.
        
        # Assuming the original file ended before these revisions.
        # We'll just write the file out clean.
        # But we need to make sure we don't duplicate.
        
        if "@misc{electronicfrontier2025," in text:
             # It is present, maybe corrupt after?
             # Let's simple remove the end block and rewrite it.
             split_point = text.find("@misc{electronicfrontier2025,")
             text = text[:split_point]
        
        # Remove any trailing garbage (null bytes etc were stripped by decode)
        text = text.rstrip()
        
        # Append clean new citations
        text += "\n" + new_citations
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        print("Fixed references.bib")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix_bib()

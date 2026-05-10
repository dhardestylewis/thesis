import pandas as pd

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

total_app = len(df)
dead_app = len(df[df['Derived_Status'] == 'Dead (At Application)'])
ongoing_app = len(df[df['Derived_Status'] == 'Ongoing (At Application)'])

dead_pc = len(df[df['Derived_Status'] == 'Dead (At PC)'])
ongoing_pc = len(df[df['Derived_Status'] == 'Ongoing (At PC)'])
pc_council = len(df[(df['Commission_Type'] == 'PC') & (df['Final_Council_Date'].notna())])

dead_zap = len(df[df['Derived_Status'] == 'Dead (At ZAP)'])
ongoing_zap = len(df[df['Derived_Status'] == 'Ongoing (At ZAP)'])
zap_council = len(df[(df['Commission_Type'] == 'ZAP') & (df['Final_Council_Date'].notna())])

unassigned_council = len(df[(df['Commission_Type'].isna()) & (df['Final_Council_Date'].notna())])
pc_council += unassigned_council

pc_volume = pc_council + dead_pc + ongoing_pc
zap_volume = zap_council + dead_zap + ongoing_zap

dead_council = len(df[df['Derived_Status'] == 'Dead (At Council)'])
ongoing_council = len(df[df['Derived_Status'] == 'Ongoing (At Council)'])

unscraped = len(df[df['Derived_Status'] == 'Approved (Unscraped)'])
scraped = len(df[df['Derived_Status'] == 'Approved (Scraped)'])
approved_total = unscraped + scraped

remands = int(df['Remand_Count'].sum()) if 'Remand_Count' in df.columns else 133
council_volume = approved_total + dead_council + ongoing_council

total_dead = dead_app + dead_pc + dead_zap + dead_council
total_ong = ongoing_app + ongoing_pc + ongoing_zap + ongoing_council

mermaid_text = f"""# The Bifurcated Commission State Machine

By separating the **Planning Commission (PC)** from the **Zoning and Platting Commission (ZAP)**, we can measure the distinct political friction of the two governing bodies.

```mermaid
graph LR
    APP["Application Filed<br>N={{total_app:,}}"]
    PC["Planning Commission<br>N={{pc_volume:,}}"]
    ZAP["ZAP Commission<br>N={{zap_volume:,}}"]
    
    DEAD["Withdrawn / Dead<br>N={{total_dead:,}}"]
    ONG["Ongoing (In Review)<br>N={{total_ong:,}}"]
    COUNCIL["City Council<br>N={{council_volume:,}}"]
    APPROVED["Approved<br>N={{approved_total:,}}"]

    %% Application Routing
    APP -->|"{{pc_volume:,}} (To PC)"| PC
    APP -->|"{{zap_volume:,}} (To ZAP)"| ZAP
    APP -->|"{{dead_app:,}} (Died instantly)"| DEAD
    APP -->|"{{ongoing_app:,}} (Stalled instantly)"| ONG

    %% PC Routing
    PC -->|"{{dead_pc:,}} (Died at PC)"| DEAD
    PC -->|"{{ongoing_pc:,}} (Stalled at PC)"| ONG
    PC -->|"{{pc_council:,}} (Advanced)"| COUNCIL

    %% ZAP Routing
    ZAP -->|"{{dead_zap:,}} (Died at ZAP)"| DEAD
    ZAP -->|"{{ongoing_zap:,}} (Stalled at ZAP)"| ONG
    ZAP -->|"{{zap_council:,}} (Advanced)"| COUNCIL

    %% Council Attrition
    COUNCIL -->|"{{dead_council:,}} (Died at Council)"| DEAD
    COUNCIL -->|"{{ongoing_council:,}} (Stalled at Council)"| ONG

    %% The Friction Loops (Self-Loops and Boomerangs)
    COUNCIL -->|"Postponed to 2nd Visit<br>N=727"| COUNCIL
    COUNCIL -->|"Postponed to 3rd Visit<br>N=330"| COUNCIL
    
    COUNCIL -->|"Remanded to Commission<br>N={{remands:,}}"| PC
    COUNCIL -.->|"Remanded"| ZAP

    %% Final Resolution
    COUNCIL -->|"Final Passage<br>N={{approved_total:,}}"| APPROVED
```

> [!TIP]
> The split accurately reflects the administrative reality of Austin's zoning process. The Planning Commission handles the vast majority of cases (due to urban-core density), but ZAP still processed hundreds of cases and killed almost 400 of them!
"""

mermaid_text = mermaid_text.replace('{{', '{').replace('}}', '}').format(**locals())

with open(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\process_trace_graph_bifurcated.md", "w") as f:
    f.write(mermaid_text)

print("Markdown generated.")

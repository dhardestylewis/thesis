import pandas as pd

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

total_app = len(df)
unr_app = len(df[df['Derived_Status'] == 'Unresolved (At Application)'])

unr_pc = len(df[df['Derived_Status'] == 'Unresolved (At PC)'])
pc_council = len(df[(df['Commission_Type'] == 'PC') & (df['Final_Council_Date'].notna())])

unr_zap = len(df[df['Derived_Status'] == 'Unresolved (At ZAP)'])
zap_council = len(df[(df['Commission_Type'] == 'ZAP') & (df['Final_Council_Date'].notna())])

unassigned_council = len(df[(df['Commission_Type'].isna()) & (df['Final_Council_Date'].notna())])
pc_council += unassigned_council

pc_volume = pc_council + unr_pc
zap_volume = zap_council + unr_zap

unr_council = len(df[df['Derived_Status'] == 'Unresolved (At Council)'])

unscraped = len(df[df['Derived_Status'] == 'Approved (Unscraped)'])
scraped = len(df[df['Derived_Status'] == 'Approved (Scraped)'])
approved_total = unscraped + scraped

council_volume = approved_total + unr_council
total_unresolved = unr_app + unr_pc + unr_zap + unr_council

mermaid_text = f"""# The Unified Commission State Machine

To remove chronological bias, we collapsed "Dead" and "Ongoing" into a single terminal attrition state: **Unresolved**.

```mermaid
graph LR
    APP["Application Filed<br>N={{total_app:,}}"]
    PC["Planning Commission<br>N={{pc_volume:,}}"]
    ZAP["ZAP Commission<br>N={{zap_volume:,}}"]
    
    UNR["Unresolved / Failed to Pass<br>N={{total_unresolved:,}}"]
    COUNCIL["City Council<br>N={{council_volume:,}}"]
    APPROVED["Approved<br>N={{approved_total:,}}"]

    %% Application Routing
    APP -->|"{{pc_volume:,}} (To PC)"| PC
    APP -->|"{{zap_volume:,}} (To ZAP)"| ZAP
    APP -->|"{{unr_app:,}} (Died instantly)"| UNR

    %% PC Routing
    PC -->|"{{unr_pc:,}} (Stalled at PC)"| UNR
    PC -->|"{{pc_council:,}} (Advanced)"| COUNCIL

    %% ZAP Routing
    ZAP -->|"{{unr_zap:,}} (Stalled at ZAP)"| UNR
    ZAP -->|"{{zap_council:,}} (Advanced)"| COUNCIL

    %% Council Attrition
    COUNCIL -->|"{{unr_council:,}} (Stalled at Council)"| UNR

    %% Final Resolution
    COUNCIL -->|"Final Passage<br>N={{approved_total:,}}"| APPROVED
```

> [!TIP]
> This mathematically prevents the chronological illusion where only 2009 cases appeared "Dead". We now have a true accounting of total municipal friction.
"""

mermaid_text = mermaid_text.replace('{{', '{').replace('}}', '}').format(**locals())

with open(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\process_trace_graph_bifurcated.md", "w") as f:
    f.write(mermaid_text)

print("Markdown generated.")

import pandas as pd

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

def assign_unified_status(row):
    has_council = pd.notna(row.get('Final_Council_Date'))
    has_approval = pd.notna(row.get('approval_date')) or pd.notna(row.get('final_date'))
    comm_type = row.get('Commission_Type')
    
    if pd.isna(row.get('application_start_date')):
        return "Unknown"
        
    if has_approval:
        if has_council:
            return "Approved (Scraped)"
        else:
            return "Approved (Unscraped)"
            
    if has_council:
        return "Unresolved (At Council)"
        
    if pd.notna(comm_type):
        if comm_type == 'Both': comm_type = 'PC'
        return f"Unresolved (At {comm_type})"
        
    return "Unresolved (At Application)"

df['Derived_Status'] = df.apply(assign_unified_status, axis=1)

print("\nNEW Unified Breakdown:")
print(df['Derived_Status'].value_counts())

df.to_csv(model_csv, index=False)

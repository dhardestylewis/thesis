import pandas as pd
concess = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Scratch\extracted_height_concessions.csv")
print("Max value cases:")
print(concess.sort_values("staff_recommended_height", ascending=False).head(8).to_string())
gt200 = (concess["staff_recommended_height"] > 200).sum()
gt100 = (concess["staff_recommended_height"] > 100).sum()
dup = concess.groupby("case_number")["staff_recommended_height"].count()
print(f"GT200ft: {gt200}, GT100ft: {gt100}")
print(f"Unique cases: {concess['case_number'].nunique()}, multi-entry cases: {(dup > 1).sum()}")

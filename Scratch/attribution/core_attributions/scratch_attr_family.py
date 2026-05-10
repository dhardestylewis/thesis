import numpy as np

data = [
    ("Deep_ERM", [4.7, 10.2, 5.4, 13.9, 9.8, 12.0, 20.4, 9.0, 14.7]),
    ("Deep_ERM", [4.8, 8.7, 5.3, 14.7, 9.5, 12.8, 18.9, 10.1, 15.2]),
    ("Deep_ERM", [3.9, 8.9, 4.6, 15.0, 10.0, 14.2, 20.7, 9.8, 12.8]),
    ("Deep_ERM", [4.8, 8.9, 4.9, 14.7, 10.0, 14.1, 19.8, 8.9, 13.9]),
    ("Deep_ERM", [4.6, 8.8, 4.8, 15.1, 9.0, 13.6, 20.2, 9.9, 13.9]),
    ("Deep_ERM", [4.9, 9.5, 5.3, 14.6, 8.7, 12.7, 21.0, 9.9, 13.3]),
    ("Deep_ERM", [5.4, 8.3, 5.0, 15.5, 9.5, 12.5, 20.9, 9.1, 13.8]),
    ("Deep_VREx", [5.7, 3.7, 2.7, 14.3, 10.0, 9.2, 24.1, 11.9, 18.4]),
    ("Deep_VREx", [6.1, 4.1, 2.4, 12.9, 8.5, 9.0, 25.2, 13.0, 18.8]),
    ("Deep_VREx", [5.7, 3.0, 2.2, 12.9, 10.1, 9.0, 25.7, 12.8, 18.5]),
    ("Deep_VREx", [5.5, 3.3, 2.7, 13.4, 9.6, 9.7, 24.6, 13.0, 18.4]),
    ("Deep_VREx", [7.2, 4.2, 2.7, 10.8, 7.4, 7.5, 27.5, 13.1, 19.5]),
    ("Deep_VREx", [7.3, 4.1, 2.9, 9.2, 5.7, 5.0, 30.2, 14.2, 21.4]),
    ("Deep_VREx", [7.2, 5.8, 3.9, 9.7, 4.8, 5.5, 28.5, 14.1, 20.5]),
    ("Trees", [12.8, 0.6, 0.4, 0.5, 0.5, 1.5, 50.1, 13.8, 19.8]),
    ("Trees", [13.8, 0.4, 0.3, 0.5, 0.2, 0.7, 42.7, 17.3, 24.0]),
    ("Trees", [8.2, 0.4, 0.1, 0.8, 1.3, 1.1, 46.9, 19.7, 21.5]),
    ("Trees", [6.9, 0.1, 0.4, 0.6, 1.1, 3.3, 27.0, 34.0, 26.6]),
    ("Trees", [5.5, 0.0, 0.2, 2.3, 1.4, 2.7, 35.7, 20.5, 31.6]),
    ("Trees", [9.8, 0.2, 0.1, 0.7, 2.1, 1.9, 37.1, 25.0, 23.3]),
    ("Trees", [11.2, 0.4, 0.2, 1.8, 2.7, 1.8, 31.1, 23.2, 27.5]),
    ("Trees", [5.8, 0.3, 0.2, 3.5, 2.6, 2.3, 34.3, 25.8, 25.2]),
    ("Trees", [7.7, 0.3, 0.1, 2.4, 3.1, 2.2, 31.2, 27.0, 25.9]),
    ("Trees", [6.8, 0.4, 0.0, 2.6, 2.1, 4.7, 36.3, 10.2, 36.9]),
    ("Trees", [20.2, 0.1, 0.1, 0.6, 0.5, 2.7, 31.0, 18.0, 26.8]),
    ("Trees", [17.8, 0.4, 0.0, 5.0, 3.7, 3.8, 32.4, 13.1, 23.8]),
    ("Trees", [10.5, 0.1, 0.6, 5.2, 2.0, 2.1, 26.1, 19.2, 34.2]),
    ("Trees", [14.4, 0.0, 0.1, 8.9, 1.0, 1.6, 28.4, 21.6, 24.1])
]

# Note: The order of columns from user is: Neighborhood Valuation, Zoning Density, Improvement Scale, Parcel Scale, Structure Age, Property Valuation, Demographics, Neighborhood Income, Housing Tenure.
cols = ["Neighborhood Valuation", "Zoning Density", "Improvement Scale", "Parcel Scale", "Structure Age", "Property Valuation", "Demographics", "Neighborhood Income", "Housing Tenure"]

deep_erm = np.array([x[1] for x in data if x[0] == "Deep_ERM"])
deep_vrex = np.array([x[1] for x in data if x[0] == "Deep_VREx"])
trees = np.array([x[1] for x in data if x[0] == "Trees"])


print(f"{'Cluster':<22} | {'Trees Avg':<10} | {'DeepAvg':<10}")
print("-" * 50)
for i, c in enumerate(cols):
    deep_mean = np.mean(np.concatenate([deep_erm[:, i], deep_vrex[:, i]]))
    tree_mean = np.mean(trees[:, i])
    print(f"{c:<22} | {tree_mean:>9.1f}% | {deep_mean:>9.1f}%")


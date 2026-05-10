import os

modules = [
    'Scripts/pipeline/advanced_spatial_modules/calc_vectors.py',
    'Scripts/pipeline/advanced_spatial_modules/engineer_neighbor_differentials.py',
    'Scripts/pipeline/advanced_spatial_modules/generate_pca_embeddings.py',
    'Scripts/pipeline/advanced_spatial_modules/engineer_ears_differentials.py',
    'Scripts/pipeline/advanced_spatial_modules/engineer_temporal_differentials.py'
]

for mod in modules:
    with open(mod, 'r') as f:
        content = f.read()
        
    lines = content.split('\n')
    new_lines = []
    in_loading = False
    
    for line in lines:
        if line.startswith('def build_'):
            new_lines.append(line.replace('():', '(petitions, tcad, cases_gdf, props=None, out_dir=r"Data/Protest_Petitions"):'))
            continue
            
        if 'print("1. Loading datasets...")' in line:
            in_loading = True
            continue
            
        if in_loading:
            if 'signed_cases =' in line or 'print(' in line:
                in_loading = False
                new_lines.append(line)
            else:
                continue
        else:
            if r'C:\Users\dhl\data\Thesis\thesis\Scratch\Spatial_Engineering' in line:
                line = line.replace(r'C:\Users\dhl\data\Thesis\thesis\Scratch\Spatial_Engineering', r'Data\Protest_Petitions')
            new_lines.append(line)
            
    with open(mod, 'w') as f:
        f.write('\n'.join(new_lines))

print("Dependency injection refactor complete.")

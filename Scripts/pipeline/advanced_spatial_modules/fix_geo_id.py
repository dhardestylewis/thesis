import os

modules = [
    'Scripts/pipeline/advanced_spatial_modules/engineer_neighbor_differentials.py',
    'Scripts/pipeline/advanced_spatial_modules/generate_pca_embeddings.py',
    'Scripts/pipeline/advanced_spatial_modules/engineer_ears_differentials.py',
    'Scripts/pipeline/advanced_spatial_modules/engineer_temporal_differentials.py'
]

for mod in modules:
    with open(mod, 'r') as f:
        content = f.read()
        
    content = content.replace("['geo_id']", ".index")
    
    with open(mod, 'w') as f:
        f.write(content)

print('Fixed geo_id index!')

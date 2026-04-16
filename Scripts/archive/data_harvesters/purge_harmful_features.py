import pandas as pd

def purge_harmful_features(df):
    """
    Automated mask to drop features explicitly flagged as actively harmful or 
    causing collinear collapse/dilution across cross-validation.
    
    This function should be injected into the data loading pipeline right before 
    creating the final X training matrix to lightly regularize tree behavior.
    """
    harmful_features = [
        # Collinear Financials (Retain one, drop the noise)
        'land_market_value', 
        'improvement_market_value',
        # Abstract Coordinates
        'latitude',
        # Highly Granular Zoning Deltas (overfitters)
        'delta_imprv_sqft_max',
        'delta_max_far',
        # Dates (overfit to split points instead of generalizing)
        'second_most_recent_sale_date',
        'most_recent_sale_date',
        'tax_year'
    ]
    
    dropped = []
    for f in harmful_features:
        if f in df.columns:
            df = df.drop(columns=[f])
            dropped.append(f)
            
    # print(f"[*] Purge Mask Executed: Stripped {len(dropped)} harmful proxy features from tensor.")
    return df

"""
Shared ID normalization utilities for TCAD parcel joins.

H0 standardized_tcad_id: 7 or 9 digits, no dash, no leading zero
LDB_2016 PID_10:         9 digits, no leading zero  → direct match to 9-digit H0
EARS account_number:     10 digits, 1 leading zero  → '0' + 9-digit H0
"""

def normalize_tcad_to_9(tcad_id):
    """Normalize any TCAD variant to 9-digit canonical form (no leading zero)."""
    if tcad_id is None:
        return None
    s = str(tcad_id).strip()
    # Skip dash-format land-rights records
    if '-' in s and len(s) > 15:
        return None
    # Remove any dashes (short dashed format like 123-456-789)
    s = s.replace('-', '')
    # Strip leading zeros, then zero-pad to 9 digits
    s = s.lstrip('0') or '0'
    return s.zfill(9)

def normalize_ears_to_9(ears_acct):
    """Normalize EARS account_number to 9-digit form for matching."""
    if ears_acct is None:
        return None
    s = str(ears_acct).strip()
    if '-' in s and len(s) > 15:
        return None
    s = s.replace('-', '').lstrip('0') or '0'
    return s.zfill(9)

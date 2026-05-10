import os
import shutil

ROOT = r'c:\Users\dhl\data\Thesis\thesis'
OUT_DIR = os.path.join(ROOT, 'Scripts', 'nlp_pipeline')
os.makedirs(OUT_DIR, exist_ok=True)

SCRIPTS = {
    r'Scratch\Data_Acquisition\download_all_commission_pdfs_v2.py': (
        '01_download_commission_pdfs.py',
        '"""\nPhase 1: Asynchronous PDF Acquisition\nDownloads 10,000+ City of Austin Council and Commission meeting agendas and transcripts.\nHandles rate-limiting and asynchronous networking to build the local Data/Commission_PDFs corpus.\n"""\n'
    ),
    r'Scratch\Misc_Scripts\transcribe_commission_pdfs.py': (
        '02_transcribe_pdfs.py',
        '"""\nPhase 2: PDF Optical Character Recognition\nIterates through the 10,000+ downloaded PDFs and uses PyMuPDF to extract text.\nOutputs the raw textual data to commission_transcripts.csv for NLP parsing.\n"""\n'
    ),
    r'Scratch\Data_Processing\parse_commission_hearings.py': (
        '03_parse_hearings.py',
        '"""\nPhase 3: NLP Entity Extraction and Regex Parsing\nScans the raw commission_transcripts.csv text to identify formal zoning case numbers \nand their associated meeting contexts, generating the base hearing log.\n"""\n'
    ),
    r'Scratch\Data_Processing\extract_temporal_zoning_v6.py': (
        '04_extract_temporal_zoning.py',
        '"""\nPhase 4: Temporal State Machine (V6)\nThe core NLP state machine. Reconstructs the chronological timeline (App_Date, Final_Council_Date,\nCouncil Appearances) by tracking how each zoning case moved through the extracted hearings.\n"""\n'
    ),
    r'Scratch\Data_Processing\calculate_remands.py': (
        '05_calculate_remands.py',
        '"""\nPhase 5: Feature Engineering (Remand Logic)\nAnalyzes the temporal timeline generated in Phase 4 to calculate structural delays and \nremand counts (the number of times a case was delayed or postponed).\n"""\n'
    )
}

for src_rel, (dest_name, docstring) in SCRIPTS.items():
    src_abs = os.path.join(ROOT, src_rel)
    dest_abs = os.path.join(OUT_DIR, dest_name)
    
    if not os.path.exists(src_abs):
        print(f'Missing: {src_abs}')
        continue
        
    with open(src_abs, 'r', encoding='utf-8') as f:
        content = f.read()
        
    with open(dest_abs, 'w', encoding='utf-8') as f:
        f.write(docstring + '\n' + content)
        
    print(f'Migrated and documented: {dest_name}')

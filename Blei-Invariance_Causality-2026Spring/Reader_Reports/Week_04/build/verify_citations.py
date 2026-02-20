import re
import os

# Configuration
TEX_FILE = "Columbia - STAT8101 Reader Report Project Proposal dl3645 20260209_155859.tex"
REF_DIR = "refs"
TERMS_TO_CHECK = [
    "causal mechanism",
    "conditional distribution",
    "autonomous",
    "invariant",
    "shift",
    "environments",
    "conditional mechanism",
    "stable",
    "ICP",
    "plausible causal predictors",
    "subsets",
    "yielding",
    "intersection",
    "identify",
    "causal ancestors",
    "anchor regression",
    "diluted causality",
    "robust",
    "worst-case risk"
]

# Map terms to likely source files (heuristics based on citations)
TERM_SOURCE_MAPPING = {
    "ICP": ["jrsssb_78_5_947.txt", "PetersJanzingSchoelkopf2018_Annotation.md"],
    "plausible causal predictors": ["jrsssb_78_5_947.txt"],
    "intersection": ["jrsssb_78_5_947.txt"],
    "anchor regression": ["Bühlmann2020_Annotation.md", "Bühlmann2020.txt"], # Assuming .txt might exist or just .md
    "causal mechanism": ["PetersJanzingSchoelkopf2018_Annotation.md"]
}

def load_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def scan_references():
    print(f"--- Starting Programmatic Verification of {len(TERMS_TO_CHECK)} Terms ---")
    
    # 1. Load Reference Content
    ref_contents = {}
    if os.path.exists(REF_DIR):
        for filename in os.listdir(REF_DIR):
            if filename.endswith(".txt") or filename.endswith(".md"):
                path = os.path.join(REF_DIR, filename)
                ref_contents[filename] = load_text(path)
                print(f"Loaded reference: {filename} ({len(ref_contents[filename])} chars)")
    else:
        print(f"CRITICAL: Reference directory '{REF_DIR}' not found.")
        return

    # 2. Check each term
    results = {}
    for term in TERMS_TO_CHECK:
        found_in = []
        term_lower = term.lower()
        
        for ref_name, content in ref_contents.items():
            if term_lower in content.lower():
                # Find context (first occurrence)
                idx = content.lower().find(term_lower)
                start = max(0, idx - 50)
                end = min(len(content), idx + 100)
                context = content[start:end].replace("\n", " ")
                found_in.append(f"{ref_name} (...{context}...)")
        
        results[term] = found_in

    # 3. Output Report
    with open("verification_report.txt", "w", encoding="utf-8") as f:
        f.write("--- Verification Results ---\n")
        verified_count = 0
        for term in TERMS_TO_CHECK:
            if results[term]:
                f.write(f"[PASSED] '{term}' found in {len(results[term])} source(s):\n")
                for s in results[term][:2]: # Show max 2 sources
                    f.write(f"  -> {s}\n")
                verified_count += 1
            else:
                f.write(f"[FAILED] '{term}' NOT found in any local source text.\n")
        
        f.write(f"\nSummary: {verified_count}/{len(TERMS_TO_CHECK)} terms verified against local knowledge base.\n")
    
    print(f"Verification complete. Results written to verification_report.txt")

if __name__ == "__main__":
    scan_references()

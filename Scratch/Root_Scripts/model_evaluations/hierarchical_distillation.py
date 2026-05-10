import re
from pathlib import Path

def parse_annotations(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    annotations = []
    current_case = "C14-2013-0104" # Default fallback
    for line in lines:
        if line.startswith('| `') or line.startswith('| 20'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 4:
                file_name = parts[0].replace('`', '')
                occ = parts[1]
                snippet = parts[2].replace('`', '')
                anno = parts[3]
                
                # Try to extract case from file name or snippet
                if "0104" in file_name or "0104" in snippet or "Shelley Tract" in snippet:
                    current_case = "C14-2013-0104"
                elif "0098" in file_name or "0098" in snippet or "Mandeville" in snippet:
                    current_case = "C14-2013-0098"
                elif "0247" in file_name or "0247" in snippet or "Buckets" in snippet or "Cesar Chavez" in file_name:
                    current_case = "C14-2008-0247"
                    
                annotations.append({
                    "case": current_case,
                    "anno": anno,
                    "signal_type": "Historical" if "Historical" in anno or "Contextual" in anno else "Signal"
                })
    return annotations

def synthesize_pair(a, b):
    # If cases don't match, we don't synthesize them together. Just return a list.
    if a['case'] != b['case']:
        return [a, b]
        
    if a['signal_type'] == 'Signal' and b['signal_type'] == 'Historical': return [a]
    if b['signal_type'] == 'Signal' and a['signal_type'] == 'Historical': return [b]
    
    if a['signal_type'] == 'Historical' and b['signal_type'] == 'Historical':
        return [{"case": a['case'], "anno": "Consolidated Historical/Contextual References", "signal_type": "Historical"}]
        
    if a['signal_type'] == 'Signal' and b['signal_type'] == 'Signal':
        # Remove duplicates
        if a['anno'] == b['anno']: return [a]
        return [{"case": a['case'], "anno": f"{a['anno']} -> {b['anno']}", "signal_type": "Signal"}]
        
    return [a]

def distill_level(annotations):
    distilled = []
    i = 0
    while i < len(annotations):
        if i + 1 < len(annotations) and annotations[i]['case'] == annotations[i+1]['case']:
            res = synthesize_pair(annotations[i], annotations[i+1])
            distilled.extend(res)
            i += 2
        else:
            distilled.append(annotations[i])
            i += 1
            
    # Quick deduplication for consecutive identical historicals
    dedup = []
    for d in distilled:
        if len(dedup) > 0 and dedup[-1]['anno'] == d['anno'] and dedup[-1]['case'] == d['case']:
            continue
        dedup.append(d)
        
    return dedup

def write_level(level_num, annotations, out_dir):
    out_path = out_dir / f"distillation_level_{level_num}.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Distillation Level {level_num}\n")
        f.write(f"Total Annotations: {len(annotations)}\n\n")
        f.write("| Case ID | Synthesized Signal |\n")
        f.write("|---------|--------------------|\n")
        for a in annotations:
            f.write(f"| `{a['case']}` | {a['anno']} |\n")

def run():
    in_path = Path(r'C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts\exhaustive_case_study_annotations.md')
    out_dir = Path(r'C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts')
    
    annos = parse_annotations(in_path)
    
    # Sort purely by case to group them
    annos = sorted(annos, key=lambda x: x['case'])
    
    l1 = distill_level(annos)
    write_level(1, l1, out_dir)
    
    l2 = distill_level(l1)
    write_level(2, l2, out_dir)
    
    l3 = distill_level(l2)
    write_level(3, l3, out_dir)
    
    l4 = distill_level(l3)
    # Just force one more to get max consolidation
    l5 = distill_level(l4)
    write_level(4, l5, out_dir)

if __name__ == "__main__":
    run()

import re
from pathlib import Path

def evaluate_coverage():
    contexts_path = Path(r'c:\Users\dhl\data\Thesis\thesis\Scratch\case_study_contexts.md')
    annotations_path = Path(r'C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts\exhaustive_case_study_annotations.md')
    
    with open(contexts_path, 'r', encoding='utf-8') as f:
        contexts_text = f.read()
        
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations_text = f.read()
        
    # Extract all files mentioned in the contexts
    files_in_contexts = re.findall(r'### File:\s*(.*)', contexts_text)
    
    missing_files = []
    annotated_files = []
    
    for file in files_in_contexts:
        if file.strip() in annotations_text:
            annotated_files.append(file)
        else:
            missing_files.append(file)
            
    print(f"Total Context Files: {len(files_in_contexts)}")
    print(f"Annotated Files: {len(annotated_files)}")
    print(f"Missing Files: {len(missing_files)}")
    print(f"Coverage: {len(annotated_files) / len(files_in_contexts) * 100:.2f}%\n")
    
    if missing_files:
        print("Missing Files List:")
        for m in missing_files:
            print(f"- {m}")

if __name__ == "__main__":
    evaluate_coverage()

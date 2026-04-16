import re

filepath = 'Austin_NIMBY_Thesis_Draft.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

def clean_short_caption(match):
    short_caption = match.group(1)
    long_caption = match.group(2)
    
    # 1. Remove \textbf{...} but keep the content inside
    short_caption = re.sub(r'\\textbf\{([^}]+)\}', r'\1', short_caption)
    # Also strip raw 'textbf' just in case
    short_caption = short_caption.replace('textbf', '')
    
    # 2. Strip trailing punctuation (:, ., ...)
    short_caption = short_caption.strip()
    while short_caption.endswith('.') or short_caption.endswith(':'):
        short_caption = short_caption[:-1].strip()
        
    return f"\\caption[{short_caption}]{{{long_caption}}}"

# Find \caption[short]{long}
new_text = re.sub(r'\\caption\[(.*?)\]\{(.*?)\}', clean_short_caption, text, flags=re.DOTALL)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Captions cleaned successfully.")

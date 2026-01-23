import re
import os

def convert_text_to_tex(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # LaTeX Preamble
    tex_content = [
        r'\documentclass[10pt, letterpaper]{article}',
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{geometry}',
        r'\geometry{top=0.75in, bottom=0.75in, left=1in, right=1in}',
        r'\usepackage{fancyhdr}',
        r'\usepackage{color}',
        r'\usepackage{xcolor}',
        r'\usepackage{titlesec}',
        r'\usepackage{enumitem}',
        r'\usepackage{hyperref}',
        r'\usepackage{amssymb}',
        r'\usepackage[framemethod=TikZ]{mdframed}',
        r'',
        r'% Official Columbia Primary Color (Dark Blue) and Tints',
        r'\definecolor{columbiablue_official}{RGB}{0, 33, 71}',
        r'\definecolor{columbiablue_light}{RGB}{240, 245, 255}',
        r'',
        r'% Response Box Environment',
        r'\newmdenv[',
        r'  backgroundcolor=columbiablue_light,',
        r'  linecolor=columbiablue_official,',
        r'  linewidth=0.5pt,',
        r'  leftline=true,',
        r'  rightline=false,',
        r'  topline=false,',
        r'  bottomline=false,',
        r'  roundcorner=2pt,',
        r'  innerleftmargin=10pt,',
        r'  innerrightmargin=10pt,',
        r'  innertopmargin=5pt,',
        r'  innerbottommargin=5pt,',
        r'  skipabove=5pt,',
        r'  skipbelow=5pt',
        r']{responsebox}',
        r'',
        r'\pagestyle{fancy}',
        r'\fancyhf{}',
        r'\renewcommand{\headrulewidth}{0.5pt}',
        r'\renewcommand{\headrule}{\hbox to\headwidth{\color{columbiablue_official}\leaders\hrule height \headrulewidth\hfill}}',
        r'',
        r'% Header',
        r'\lhead{\small \textbf{\color{columbiablue_official} COLUMBIA UNIVERSITY} \\ Graduate School of Architecture, Planning and Preservation}',
        r'\rhead{\small IRB Protocol Submission}',
        r'',
        r'\titleformat{\section}{\large\bfseries\color{columbiablue_official}}{}{0em}{}[\color{columbiablue_official}\titlerule]',
        r'\titlespacing*{\section}{0pt}{12pt}{6pt}',
        r'',
        r'\begin{document}',
        r'',
        r'\vspace*{0.2cm}',
        r'\begin{center}',
        r'    {\Large \textbf{\color{columbiablue_official} Human Subjects Protocol Data Sheet}} \\[0.5em]',
        r'    {\large Protocol \# ACYY0820 (Predicting NIMBYism)}',
        r'\end{center}',
        r'',
        r'\vspace{0.3cm}',
        r''
    ]

    known_sections = [
        "General Information", "Attributes", "Background", "Study Purpose and Rationale",
        "Study Design", "Statistical Procedures", "Exempt and Expedited", "Funding",
        "Locations", "Personnel", "Training and COI", "Departmental Approvers",
        "Privacy & Data Security", "Recruitment And Consent", "Research Aims & Abstracts",
        "Risks, Benefits & Monitoring", "Potential Risks", "Potential Benefits", "Alternatives",
        "Data and Safety Monitoring", "Subjects", "Target Enrollment Demographics", 
        "Vulnerable Populations as per 45 CFR 46", "Attached Consent Forms", "Documents", "Tasks"
    ]
    
    chars_to_escape = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'
    }

    cleaned_lines = []
    
    in_verbatim = False
    in_response_box = False
    
    # Pre-process lines to remove "Page X of Y" footers and form feeds
    lines = [l.replace('\x0c', '') for l in lines]
    lines = [re.sub(r'IRB-ACYY0820\s+Page \d+ of \d+', '', l) for l in lines]

    for line in lines:
        raw_line = line
        stripped = line.strip()
        
        # Skip purely empty lines if they don't mean paragraph breaks
        # But we do want some spacing.
        if not stripped:
            if in_verbatim:
                cleaned_lines.append("")
                continue
            if in_response_box:
                # Ensure we close response box if we hit a double gap or just end it?
                # Actually, usually response paragraphs have blank lines.
                # Let's keep it open unless we hit a new header or unindented line.
                cleaned_lines.append(r'\vspace{0.5em}') 
            else:
                cleaned_lines.append(r'\vspace{0.5em}')
            continue

        # HEADER DETECTION
        is_header = False
        upper_stripped = stripped.upper()
        # Remove trailing colon for check
        check_stripped = stripped.rstrip(':')
        
        # Exact match or Case-insensitive match against known sections
        matched_sec = None
        for sec in known_sections:
            if check_stripped == sec or check_stripped == sec.upper() or stripped == sec:
                matched_sec = sec
                break
        
        if matched_sec:
            # New Section found
            if in_verbatim:
                cleaned_lines.append(r'\end{verbatim}')
                cleaned_lines.append(r'}')
                in_verbatim = False
            if in_response_box:
                cleaned_lines.append(r'\end{responsebox}')
                in_response_box = False
            
            # Escape header text
            safe_sec = matched_sec
            for k, v in chars_to_escape.items():
                safe_sec = safe_sec.replace(k, v)
            
            cleaned_lines.append(r'\section*{' + safe_sec + '}')
            
            # Check if entering Verbatim Mode (Tables)
            if matched_sec in ["Personnel", "Locations", "Attached Consent Forms", "Documents", "Departmental Approvers", "Target Enrollment Demographics"]:
                cleaned_lines.append(r'{\footnotesize')
                cleaned_lines.append(r'\begin{verbatim}')
                in_verbatim = True
            continue

        # If in Verbatim, just dump line
        if in_verbatim:
            cleaned_lines.append(raw_line.rstrip())
            continue

        # ESCAPE LINE CONTENT
        safe_line = raw_line
        # Determine indentation BEFORE stripping
        indent_len = len(safe_line) - len(safe_line.lstrip())
        
        # Escape chars
        processed_line = safe_line.strip()
        for k, v in chars_to_escape.items():
            processed_line = processed_line.replace(k, v)

        # CHECKBOX HANDLING
        has_checkbox = False
        if '[x]' in processed_line:
            processed_line = processed_line.replace('[x]', r'\makebox[0pt][l]{\color{columbiablue_official}$\boxtimes$}\hspace{1.2em}')
            has_checkbox = True
        elif '[ ]' in processed_line:
            processed_line = processed_line.replace('[ ]', r'\makebox[0pt][l]{$\square$}\hspace{1.2em}')
            has_checkbox = True

        # RESPONSE DETECTION logic
        # If line is significantly indented (> 3 spaces) AND NOT a checkbox line
        # It is likely a user response block.
        if indent_len > 3 and not has_checkbox:
            if not in_response_box:
                cleaned_lines.append(r'\begin{responsebox}')
                in_response_box = True
            cleaned_lines.append(processed_line)
            continue
        else:
            # Unindented or Checkbox
            if in_response_box:
                cleaned_lines.append(r'\end{responsebox}')
                in_response_box = False

        # Apply Key-Value Bolding for "Prompt" lines
        if ':' in processed_line:
            parts = processed_line.split(':', 1)
            # Heuristic: Key must be short (< 60 chars) and not contain latex macros (starts with \)
            if len(parts[0]) < 60 and '\\' not in parts[0] and not has_checkbox:
                processed_line = r'\noindent \textbf{' + parts[0] + r':}' + parts[1]
        
        # If it's a checkbox line, ensure strict spacing/noindent
        if has_checkbox:
            processed_line = r'\noindent ' + processed_line

        # If it looks like an instruction "Provide..." or "Select...", maybe italicize?
        # Only if unindented and long-ish? safe assumption: regular text.
        # But let's add `\\` for line break preservation if needed.
        # Actually normal paragraph flow is better for long instructions.
        # But converting text to latex requires handling newlines.
        # We will append blank lines for spacing, so simple text is fine.
        
        cleaned_lines.append(processed_line)

    # Close any open blocks at end
    if in_verbatim:
        cleaned_lines.append(r'\end{verbatim}')
    if in_response_box:
        cleaned_lines.append(r'\end{responsebox}')

    tex_content.append("\n\n".join(cleaned_lines))
    tex_content.append(r'\end{document}')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_content))

if __name__ == "__main__":
    convert_text_to_tex(
        r"c:\Users\dhl\data\thesis\thesis\Submitted\IRB_Submitted\Protocol_and_Submission\Original_Source\IRB_Submitted_Protocol.txt",
        r"c:\Users\dhl\data\thesis\thesis\Submitted\IRB_Submitted\Protocol_and_Submission\Generated_PDF\IRB_Submitted_Protocol.tex"
    )

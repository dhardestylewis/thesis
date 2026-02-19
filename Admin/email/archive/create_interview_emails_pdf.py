import os
import email
from email import policy
from fpdf import FPDF

def extract_email_content(file_path):
    with open(file_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
    
    subject = msg['subject']
    from_addr = msg['from']
    to_addr = msg['to']
    date = msg['date']
    
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_content()
                except:
                    pass
                break # Prefer text/plain
    else:
        try:
            body = msg.get_content()
        except:
            pass
            
    return {
        "Subject": subject,
        "From": from_addr,
        "To": to_addr,
        "Date": date,
        "Body": body
    }

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Interview / Outreach Emails', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f'Email {num}: {label}', 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        # Sanitize for FPDF (latin-1 only by default)
        body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 5, body)
        self.ln()

def create_pdf(email_files, output_filename):
    pdf = PDF()
    pdf.add_page()
    
    for i, file_path in enumerate(email_files):
        try:
            content = extract_email_content(file_path)
            title = content['Subject']
            text = f"From: {content['From']}\nTo: {content['To']}\nDate: {content['Date']}\n\n{content['Body']}"
            
            pdf.chapter_title(i + 1, title)
            pdf.chapter_body(text)
            pdf.add_page()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    pdf.output(output_filename)
    print(f"Created {output_filename}")

if __name__ == "__main__":
    # Correct paths based on previous list_dir
    base_dir = r"C:\Users\dhl\.gemini\antigravity\scratch\gmail-mcp-workspace\emails_verbatim"
    files = [
        os.path.join(base_dir, "19bee26d0c2131c2.eml"),
        os.path.join(base_dir, "19b99a9f63efa9ed.eml")
    ]
    
    output_pdf = r"c:\Users\dhl\data\thesis\thesis\Interview_Emails.pdf"
    create_pdf(files, output_pdf)

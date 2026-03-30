import time
import pandas as pd
import os
import ollama

def main():
    ROOT = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV"
    pdf_path = os.path.join(ROOT, "scraped_backup_pdf_links.csv")
    txt_path = os.path.join(ROOT, "scraped_agenda_text_embeddings.csv")
    
    with open("out_metrics.txt", "w", encoding='utf-8') as f:
        f.write("--- PDF BACKUP SUCCESS METRICS ---\n")
        df = pd.read_csv(pdf_path)
        docs = df['document_count']
        has_pdfs = (docs > 0).sum()
        total = len(df)
        pct = (has_pdfs / total) * 100
        total_pdfs = docs.sum()
        f.write(f"Total Cases Tracked: {total}\n")
        f.write(f"Cases with Backup PDFs Found: {has_pdfs} ({pct:.1f}%)\n")
        f.write(f"Total individual PDF URLs harvested: {total_pdfs}\n")
        
        f.write("\n--- EMBEDDING TIME ESTIMATE ---\n")
        df_txt = pd.read_csv(txt_path)
        texts = df_txt['agenda_text_raw'].dropna().tolist()
        num_texts = len(texts)
        f.write(f"Total raw text blocks to embed: {num_texts}\n")
        
        f.write("Running local Ollama embedding benchmark (model='llama3.1:8b')...\n")
        
        sample_size = 10
        sample_texts = [str(x)[:1000] for x in texts[:sample_size]]
        
        t0 = time.time()
        success = False
        try:
            for txt in sample_texts:
                ollama.embeddings(model='llama3.1:8b', prompt=txt)
            success = True
        except Exception as e:
            f.write(f"Ollama failed: {e}\n")
            
        t1 = time.time()
        elapsed = t1 - t0
        
        if success:
            avg_per_doc = elapsed / sample_size
            est_total_secs = avg_per_doc * num_texts
            
            f.write(f"\nEmbedded {sample_size} documents in {elapsed:.2f} seconds.\n")
            f.write(f"Speed: {avg_per_doc:.2f} seconds per document.\n")
            f.write(f"ESTIMATED TOTAL TIME FOR {num_texts} DOCS: {est_total_secs/60:.2f} minutes.\n")

if __name__ == "__main__":
    main()

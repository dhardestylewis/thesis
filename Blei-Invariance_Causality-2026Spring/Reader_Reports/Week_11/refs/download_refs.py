"""
Download PDFs and extract text for Week 11 references.
Pulls from arXiv (open access) and gaussianprocess.org.
"""

import urllib.request
import os
import time
from pypdf import PdfReader

REFS_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {'User-Agent': 'mailto:dhl2147@columbia.edu'}

PAPERS = [
    {
        "key": "Srinivas2010_GP_UCB",
        "title": "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design",
        "authors": "Srinivas, Krause, Kakade, Seeger",
        "year": 2010,
        "url": "https://arxiv.org/pdf/0912.3995",
    },
    {
        "key": "Wang2013_REMBO",
        "title": "Bayesian Optimization in a Billion Dimensions via Random Embeddings",
        "authors": "Wang, Zoghi, Hutter, Matheson, de Freitas",
        "year": 2013,
        "url": "https://arxiv.org/pdf/1301.1942",
    },
    {
        "key": "Eriksson2019_TuRBO",
        "title": "Scalable Global Optimization via Local Bayesian Optimization (TuRBO)",
        "authors": "Eriksson, Pearce, Gardner, Turner, Poloczek",
        "year": 2019,
        "url": "https://arxiv.org/pdf/1910.01739",
    },
    {
        "key": "Wilson2016_DeepKernel",
        "title": "Deep Kernel Learning",
        "authors": "Wilson, Hu, Salakhutdinov, Xing",
        "year": 2016,
        "url": "https://arxiv.org/pdf/1511.02222",
    },
    {
        "key": "Russo2014_IDS",
        "title": "Learning to Optimize via Information-Directed Sampling",
        "authors": "Russo, Van Roy",
        "year": 2014,
        "url": "https://arxiv.org/pdf/1403.5556",
    },
    {
        "key": "Letham2019_ConstrainedBO",
        "title": "Constrained Bayesian Optimization with Noisy Experiments",
        "authors": "Letham, Karrer, Ottoni, Bakshy",
        "year": 2019,
        "url": "https://arxiv.org/pdf/1706.07094",
    },
    {
        "key": "Rasmussen2006_GP",
        "title": "Gaussian Processes for Machine Learning",
        "authors": "Rasmussen, Williams",
        "year": 2006,
        "url": "http://gaussianprocess.org/gpml/chapters/RW.pdf",
    },
]


def download_and_extract(paper):
    pdf_path = os.path.join(REFS_DIR, paper["key"] + ".pdf")
    txt_path = os.path.join(REFS_DIR, paper["key"] + ".txt")

    if os.path.exists(pdf_path):
        print(f"  Already downloaded: {paper['key']}")
    else:
        print(f"  Downloading {paper['key']} ...")
        try:
            req = urllib.request.Request(paper["url"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(pdf_path, "wb") as f:
                    f.write(resp.read())
            print(f"  Saved {pdf_path}")
            time.sleep(1)
        except Exception as e:
            print(f"  FAILED to download {paper['key']}: {e}")
            return False

    if not os.path.exists(txt_path):
        try:
            reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  Extracted text to {txt_path}")
        except Exception as e:
            print(f"  FAILED to extract text from {paper['key']}: {e}")

    return True


if __name__ == "__main__":
    for paper in PAPERS:
        print(f"\n[{paper['key']}]")
        download_and_extract(paper)
    print("\nDone.")

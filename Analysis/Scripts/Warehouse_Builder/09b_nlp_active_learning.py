import os
import glob
import pandas as pd
import numpy as np
import ollama
import json
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
TRANSCRIPTS_DIR = os.path.join(ROOT_DIR, "Data", "Zoning_Cases", "Processed_Data", "Transcripts")
WORK_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of", "Build")
seed_size = 50

def parse_llm_response(resp_text):
    # Fallback parser if json decode fails
    resp_text = resp_text.lower()
    frames = {
        'frame_traffic': 1 if 'traffic' in resp_text else 0,
        'frame_infrastructure': 1 if 'infrastructure' in resp_text else 0,
        'frame_displacement': 1 if 'displacement' in resp_text else 0,
        'frame_neighborhood': 1 if 'character' in resp_text or 'neighborhood' in resp_text else 0,
        'frame_procedural': 1 if 'fairness' in resp_text or 'process' in resp_text else 0
    }
    stance = 'neutral'
    if 'oppose' in resp_text: stance = 'oppose'
    elif 'support' in resp_text: stance = 'support'
    
    return stance, frames

def generate_active_learning_labels():
    print(f"Loading transcribed text files from {TRANSCRIPTS_DIR}...")
    txt_files = glob.glob(os.path.join(TRANSCRIPTS_DIR, "*_transcript.txt"))
    
    records = []
    for fpath in txt_files:
        filename = os.path.basename(fpath)
        case_number = filename.replace("_transcript.txt", "")
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
                if len(text) > 50:
                    records.append({"CASE_NUMBER": case_number, "text": text})
        except Exception:
            pass
            
    raw_df = pd.DataFrame(records)
    print(f"Loaded {len(raw_df)} cases.")
    if len(raw_df) == 0: return
    
    np.random.seed(42)
    # Shuffle
    raw_df = raw_df.sample(frac=1).reset_index(drop=True)
    
    seed_df = raw_df.iloc[:seed_size]
    unlabeled_df = raw_df.iloc[seed_size:]
    
    print(f"Starting LLM Zero-Shot Pseudo-Labeling on {seed_size} samples using Llama 3.2:3b...")
    
    stances = []
    frames_list = []
    
    for i, row in seed_df.iterrows():
        text_snippet = row['text'][:1500] # Fit context window
        prompt = f"""Analyze the following public hearing zoning transcript speech segment.
1. Determine the overall stance (Support, Oppose, Neutral) regarding the proposed zoning change.
2. Identify which of the following argument frames are present: [Traffic, Infrastructure, Displacement, Neighborhood Character, Procedural Fairness].
Return ONLY a valid JSON object with keys "stance" and a list "frames".

Transcript:
{text_snippet}
"""
        try:
            res = ollama.chat(model='llama3.2:3b', messages=[{'role': 'user', 'content': prompt}])
            resp_content = res['message']['content']
            stance, frames = parse_llm_response(resp_content)
        except Exception as e:
            print("Ollama Error on segment", i, e)
            stance = 'neutral'
            frames = {'frame_traffic':0, 'frame_infrastructure':0, 'frame_displacement':0, 'frame_neighborhood':0, 'frame_procedural':0}
            
        stances.append(stance)
        frames_list.append(frames)
        
    seed_df['true_stance'] = stances
    frames_df = pd.DataFrame(frames_list)
    for c in frames_df.columns:
        seed_df[c] = frames_df[c].values
        
    # Active Learning / Expansion Phase
    print("Expanding coverage iteratively across all transcripts via Scikit-Learn TF-IDF classification...")
    
    X_train = seed_df['text']
    X_all = raw_df['text']
    
    # Train predictors
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Predict Stance
    clf_stance = make_pipeline(vectorizer, LogisticRegression(class_weight='balanced'))
    clf_stance.fit(X_train, seed_df['true_stance'])
    stance_preds = clf_stance.predict(X_all)
    stance_probs = clf_stance.predict_proba(X_all)
    stances_classes = clf_stance.classes_
    
    raw_df['stance_pred'] = stance_preds
    if 'oppose' in stances_classes:
        opp_idx = list(stances_classes).index('oppose')
        raw_df['has_transcribed_opposition'] = stance_probs[:, opp_idx] > 0.4
    else:
        raw_df['has_transcribed_opposition'] = 0
        
    raw_df['has_transcribed_opposition'] = raw_df['has_transcribed_opposition'].astype(int)

    # Predict Frames
    frame_cols = [c for c in seed_df.columns if c.startswith('frame_')]
    for col in frame_cols:
        clf_frame = make_pipeline(vectorizer, LogisticRegression(class_weight='balanced'))
        # If there's only one class predicted by LLM, fallback
        if len(seed_df[col].unique()) > 1:
            clf_frame.fit(X_train, seed_df[col])
            raw_df[f'prob_{col}'] = clf_frame.predict_proba(X_all)[:, 1]
        else:
            raw_df[f'prob_{col}'] = seed_df[col].iloc[0] # All 0 or 1
            
    # For backwards compatibility with plot_F19_F20, let's also export TF-IDF unigrams 
    # but the new plot script will explicitly use prob_frame_*!
    # For now, we will add dummy tfidf columns if needed, but preferably we just update F19/F20.
    
    # We will export the final CSV
    out_cols = ['CASE_NUMBER', 'has_transcribed_opposition'] + [f'prob_{c}' for c in frame_cols]
    final_df = raw_df[out_cols]
    
    out_path = os.path.join(WORK_DIR, "speech_comment.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Active Learning Expansion complete. Exported `{out_path}`.")

if __name__ == "__main__":
    generate_active_learning_labels()

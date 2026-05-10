import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

transcripts_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv"
model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"

df_trans = pd.read_csv(transcripts_csv)
df_model = pd.read_csv(model_csv)

analyzer = SentimentIntensityAnalyzer()

def extract_case_number(filename):
    m = re.search(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', str(filename).upper())
    return m.group(1) if m else None

df_trans['Core_Case'] = df_trans['Filename'].apply(extract_case_number)

features = []

print("Extracting NLP and Regex Features from Commission Transcripts...")

for case_num, group in df_trans.groupby('Core_Case'):
    if not case_num:
        continue
        
    case_data = {
        'Core_Case': case_num,
        'Staff_Recommendation': None,
        'label_valid_petition_pct': 0.0,
        'Opposition_Volume': 0,
        'Support_Volume': 0,
        'Aggregate_Sentiment': 0.0,
        'Primary_Complaint': 'None'
    }
    
    sentiment_scores = []
    complaint_counts = {'Traffic': 0, 'Density': 0, 'Environment': 0, 'Gentrification': 0}
    
    for _, row in group.iterrows():
        text = str(row['Raw_Text'])
        filename = str(row['Filename']).lower()
        
        # Determine if Staff Report
        if 'staff' in filename or 'report' in filename or 'recommendation' in text.lower():
            # Regex for Valid Petition
            pet_match = re.search(r'Valid\s+Petition[^\d]*?(\d{1,3}(?:\.\d+)?)%', text, re.IGNORECASE)
            if pet_match:
                case_data['label_valid_petition_pct'] = float(pet_match.group(1))
            
            # Regex for Staff Recommendation
            rec_match = re.search(r'Staff Recommendation:\s*(Approval|Denial|approval|denial)', text, re.IGNORECASE)
            if rec_match:
                case_data['Staff_Recommendation'] = rec_match.group(1).capitalize()
                
        # Determine if Public Comment / Opposition
        if 'opposition' in filename or 'petition' in filename or 'letter' in filename or 'public' in filename or 'comment' in filename or ('staff' not in filename and 'report' not in filename):
            if 'support' in filename or 'favor' in text.lower()[:500]:
                case_data['Support_Volume'] += 1
            else:
                case_data['Opposition_Volume'] += 1
                
                score = analyzer.polarity_scores(text)['compound']
                sentiment_scores.append(score)
                
                text_lower = text.lower()
                complaint_counts['Traffic'] += text_lower.count('traffic') + text_lower.count('parking') + text_lower.count('congestion')
                complaint_counts['Density'] += text_lower.count('density') + text_lower.count('height') + text_lower.count('character') + text_lower.count('scale')
                complaint_counts['Environment'] += text_lower.count('tree') + text_lower.count('flood') + text_lower.count('runoff') + text_lower.count('environment')
                complaint_counts['Gentrification'] += text_lower.count('tax') + text_lower.count('affordability') + text_lower.count('gentrification') + text_lower.count('displace')

    if sentiment_scores:
        case_data['Aggregate_Sentiment'] = sum(sentiment_scores) / len(sentiment_scores)
        
    if sum(complaint_counts.values()) > 0:
        case_data['Primary_Complaint'] = max(complaint_counts, key=complaint_counts.get)
        
    features.append(case_data)

df_features = pd.DataFrame(features)

# Merge back into model_csv
df_final = pd.merge(df_model, df_features, on='Core_Case', how='left')

# Fill NaNs
df_final['label_valid_petition_pct'] = df_final['label_valid_petition_pct'].fillna(0.0)
df_final['Opposition_Volume'] = df_final['Opposition_Volume'].fillna(0)
df_final['Support_Volume'] = df_final['Support_Volume'].fillna(0)
df_final['Aggregate_Sentiment'] = df_final['Aggregate_Sentiment'].fillna(0.0)
df_final['Primary_Complaint'] = df_final['Primary_Complaint'].fillna('None')

df_final.to_csv(model_csv, index=False)
print(f"Finished! Merged Commission NLP/Regex features for {len(df_features)} cases.")
print(f"Target Master File: {model_csv}")

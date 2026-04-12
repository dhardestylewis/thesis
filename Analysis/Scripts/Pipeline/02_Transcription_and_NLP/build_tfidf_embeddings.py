"""
This module provides a rolling TF-IDF and SVD extraction pipeline.
Using this class prevents the data leakage inherent in fitting global topic embeddings
on future unseen documents.

It should be imported and used dynamically during the cross-validation
or chronological backtest of the main predictive models.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TimeAwareTextEmbedder:
    def __init__(self, max_features=1000, max_components=20, random_state=42):
        self.max_features = max_features
        self.max_components = max_components
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        self.svd = None
        self.n_components = None
        
    def fit(self, train_texts):
        """Fits vocabulary and latent space strictly on the training window."""
        # Ensure we don't crash on completely empty or NA texts
        clean_texts = train_texts.fillna("").astype(str)
        
        try:
            tfidf_matrix = self.tfidf.fit_transform(clean_texts)
            # Guard against degenerate dimensions if the training set is very small
            vocab_size = tfidf_matrix.shape[1]
            self.n_components = min(self.max_components, vocab_size - 1)
        except ValueError:
            # Empty vocabulary (all stop words or empty strings)
            self.n_components = 0
            
        if self.n_components is not None and self.n_components > 0:
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
            self.svd.fit(tfidf_matrix)

    def transform(self, texts):
        """Applies the learned vocabulary and decomposition to arbitrary text (train or test)."""
        if not self.n_components or self.n_components == 0:
            return pd.DataFrame(index=texts.index)
            
        clean_texts = texts.fillna("").astype(str)
        tfidf_matrix = self.tfidf.transform(clean_texts)
        
        if self.n_components and self.n_components > 0 and self.svd:
            embed_matrix = self.svd.transform(tfidf_matrix)
            df_out = pd.DataFrame(embed_matrix, columns=[f'nlp_svd_{i}' for i in range(self.n_components)])
            return df_out
        else:
            return pd.DataFrame(index=texts.index)
        
    def fit_transform(self, train_texts):
        self.fit(train_texts)
        return self.transform(train_texts)

if __name__ == '__main__':
    print("This script is now a module for rolling window evaluation!")
    print("WARNING: Do NOT run this globally. Import TimeAwareTextEmbedder into your backtest script.")

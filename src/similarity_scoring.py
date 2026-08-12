import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_and_save_tfidf(preprocessed_resumes, save_path=r"D:\FUTURE_ML_O3\models\tfidf_vectorizer.pkl"):
    """
    Fits a TfidfVectorizer on all cleaned resumes and serializes it using joblib.
    """
    print("[SimilarityScoring] Initializing TfidfVectorizer...")
    
    # Configure vectorizer with standard ngram ranges and min_df to reduce vocabulary noise
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b|\b\w+\+\+\b|\b\w+\#\b" # Custom pattern to capture C++ and C# correctly
    )
    
    print("[SimilarityScoring] Fitting vectorizer on resume corpus...")
    vectorizer.fit(preprocessed_resumes)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    print(f"[SimilarityScoring] Vectorizer saved to: {save_path}")
    
    return vectorizer

def load_tfidf_vectorizer(load_path=r"D:\FUTURE_ML_O3\models\tfidf_vectorizer.pkl"):
    """
    Loads a serialized TfidfVectorizer.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"TfidfVectorizer not found at {load_path}. Please train it first.")
    return joblib.load(load_path)

def calculate_cosine_similarity(vectorizer, preprocessed_resumes, preprocessed_jd):
    """
    Transforms text and computes Cosine Similarity between resumes and a single job description.
    
    Parameters:
    vectorizer (TfidfVectorizer): Pre-trained TF-IDF vectorizer.
    preprocessed_resumes (list/pd.Series): Preprocessed resume strings.
    preprocessed_jd (str): Preprocessed job description.
    
    Returns:
    np.ndarray: Vector of similarity percentage scores.
    """
    # Transform resumes to TF-IDF matrix
    resume_matrix = vectorizer.transform(preprocessed_resumes)
    
    # Transform job description to single TF-IDF vector
    jd_vector = vectorizer.transform([preprocessed_jd])
    
    # Calculate Cosine Similarity
    similarities = cosine_similarity(resume_matrix, jd_vector).flatten()
    
    # Convert to percentage
    similarity_percentages = similarities * 100.0
    
    return np.round(similarity_percentages, 2)

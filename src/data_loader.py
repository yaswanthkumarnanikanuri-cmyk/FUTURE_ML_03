import os
import pandas as pd
import numpy as np

def load_resume_dataset(filepath=r"D:\FUTURE_ML_O3\data\Resume.csv"):
    """
    Loads and cleans the raw Kaggle Resume dataset.
    Generates anonymous candidate IDs and handles missing values.
    
    Parameters:
    filepath (str): Path to the Resume.csv file.
    
    Returns:
    pd.DataFrame: Cleaned resume dataframe with CAND_ID and clean fields.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Resume dataset not found at {filepath}. Please run the downloader first.")
        
    print(f"[DataLoader] Reading dataset from: {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[DataLoader] Initial dataset shape: {df.shape}")
    
    # 1. Deduplicate records
    initial_len = len(df)
    df = df.drop_duplicates(subset=["Resume_str"])
    print(f"[DataLoader] Removed {initial_len - len(df)} duplicate records based on resume text.")
    
    # 2. Handle missing values
    null_resumes = df["Resume_str"].isnull().sum()
    if null_resumes > 0:
        print(f"[DataLoader] Dropping {null_resumes} records with empty Resume_str.")
        df = df.dropna(subset=["Resume_str"])
        
    # 3. Create anonymous candidate IDs if not already present
    # Format: CAND_001, CAND_002, etc.
    df = df.reset_index(drop=True)
    df["CAND_ID"] = [f"CAND_{i+1:03d}" for i in range(len(df))]
    
    # Check category distribution
    print(f"[DataLoader] Cleaned dataset size: {df.shape}")
    print(f"[DataLoader] Unique job categories in dataset: {df['Category'].nunique()}")
    
    return df

if __name__ == "__main__":
    try:
        data = load_resume_dataset()
        print(data.head(2))
    except Exception as e:
        print("Error:", e)

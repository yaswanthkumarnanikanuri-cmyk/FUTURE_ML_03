import os
import pandas as pd
import numpy as np
from src.data_loader import load_resume_dataset
from src.text_preprocessing import preprocess_text
from src.skill_extraction import extract_skills_from_text
from src.job_parser import parse_job_description
from src.similarity_scoring import train_and_save_tfidf, calculate_cosine_similarity
from src.skill_gap_analysis import perform_skill_gap_analysis
from src.candidate_ranking import compute_final_scores_and_rank, save_ranking_outputs
from src.visualization import generate_eda_plots, generate_screening_plots

def run_ml_pipeline():
    print("==================================================================")
    print("       STARTING RESUME SCREENING & RANKING ML PIPELINE           ")
    print("==================================================================")
    
    # 1. Load Resume Dataset
    df = load_resume_dataset()
    
    # To prevent slow execution on local environment during setup, we will process the dataset.
    # NLTK splitting is very fast, but let's process the full dataset to ensure real results.
    print("\n[Pipeline] Step 1: Preprocessing resumes (cleaning, tokenization, lemmatization)...")
    preprocessed_texts = []
    extracted_skills_list = []
    
    total_records = len(df)
    for i, row in df.iterrows():
        raw_text = row["Resume_str"]
        # Preprocess text
        prep_text = preprocess_text(raw_text)
        preprocessed_texts.append(prep_text)
        
        # Extract skills
        skills = extract_skills_from_text(raw_text)
        extracted_skills_list.append(skills)
        
        if (i + 1) % 500 == 0 or (i + 1) == total_records:
            print(f"Processed {i + 1}/{total_records} resumes...")
            
    df["Preprocessed_Resume"] = preprocessed_texts
    df["Extracted_Skills"] = [list(s) for s in extracted_skills_list]
    df["Skill_Count"] = df["Extracted_Skills"].apply(len)
    
    # Save a cache of the preprocessed dataset for faster loading in the Streamlit app
    cache_path = os.path.join(os.path.dirname(__file__), "data", "preprocessed_resumes.csv")
    df.to_csv(cache_path, index=False)
    print(f"[Pipeline] Preprocessed cache saved to: {cache_path}")
    
    # 2. Train and Save TF-IDF Vectorizer
    vectorizer = train_and_save_tfidf(df["Preprocessed_Resume"])
    
    # 3. Generate Exploratory Plots
    print("\n[Pipeline] Step 2: Generating EDA Plots...")
    generate_eda_plots(df, extracted_skills_list)
    
    # 4. Load a default Job Description for baseline pipeline testing
    print("\n[Pipeline] Step 3: Loading baseline Job Description...")
    jd_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "job_descriptions.csv"))
    mle_jd = jd_df[jd_df["Job_Title"] == "Machine Learning Engineer"].iloc[0]["Job_Description"]
    
    # Parse baseline JD
    parsed_jd = parse_job_description(mle_jd)
    required_skills = parsed_jd["required_skills"]
    preprocessed_jd = preprocess_text(parsed_jd["clean_text"])
    
    print("Baseline JD Title: Machine Learning Engineer")
    print("Required Skills Identified:", list(required_skills))
    
    # 5. Compute TF-IDF Cosine Similarity
    print("\n[Pipeline] Step 4: Computing Cosine Similarity scores...")
    similarities = calculate_cosine_similarity(vectorizer, df["Preprocessed_Resume"], preprocessed_jd)
    
    # 6. Perform Skill Matching & Skill Gap Analysis
    print("[Pipeline] Step 5: Performing Skill Gap analysis...")
    skill_gap_results = []
    for idx, row in df.iterrows():
        cand_skills = set(row["Extracted_Skills"])
        gap_res = perform_skill_gap_analysis(cand_skills, required_skills)
        skill_gap_results.append(gap_res)
        
    # 7. Compute Final Fit Scores and Rank Candidates
    print("[Pipeline] Step 6: Ranking candidates...")
    df_ranked = compute_final_scores_and_rank(df, similarities, skill_gap_results)
    
    # 8. Save CSV Outputs
    save_ranking_outputs(df_ranked)
    
    # 9. Generate Screening Performance Plots
    print("\n[Pipeline] Step 7: Generating screening performance plots...")
    generate_screening_plots(df_ranked)
    
    print("\n==================================================================")
    print("          PIPELINE SUCCESSFULLY COMPLETED & TESTED               ")
    print("==================================================================")
    
if __name__ == "__main__":
    run_ml_pipeline()

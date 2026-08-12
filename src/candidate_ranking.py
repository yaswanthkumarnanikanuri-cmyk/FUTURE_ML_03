import os
import pandas as pd

def compute_final_scores_and_rank(df, similarities, skill_match_results, sim_weight=0.6, skill_weight=0.4):
    """
    Combines textual similarities and skill match percentages, ranks candidates, 
    and generates an explainable fit justification.
    
    Parameters:
    df (pd.DataFrame): Dataframe with CAND_ID and Category.
    similarities (list/np.ndarray): Similarity percentages.
    skill_match_results (list of dicts): Matched/missing skills list.
    sim_weight (float): Weights for text similarity (default 0.6).
    skill_weight (float): Weights for skill match percentage (default 0.4).
    
    Returns:
    pd.DataFrame: Sorted ranking dataframe with justifications.
    """
    df_ranked = df.copy()
    
    df_ranked["Similarity_Score"] = similarities
    df_ranked["Skill_Match_Score"] = [r["skill_match_pct"] for r in skill_match_results]
    df_ranked["Matched_Skills"] = [", ".join(r["matched_skills"]) for r in skill_match_results]
    df_ranked["Missing_Skills"] = [", ".join(r["missing_skills"]) for r in skill_match_results]
    
    # Calculate Final Score
    df_ranked["Final_Score"] = round((sim_weight * df_ranked["Similarity_Score"]) + (skill_weight * df_ranked["Skill_Match_Score"]), 2)
    
    # Generate explainable candidate-fit summaries
    justifications = []
    for idx, row in df_ranked.iterrows():
        cand_id = row["CAND_ID"]
        category = row["Category"]
        sim = row["Similarity_Score"]
        skill_match = row["Skill_Match_Score"]
        matched_str = row["Matched_Skills"]
        missing_str = row["Missing_Skills"]
        
        # Build explanation
        if sim > 50 and skill_match > 50:
            just = (f"Candidate {cand_id} is an excellent fit. They show high textual similarity ({sim}%) "
                    f"to the role and possess key required skills: [{matched_str}].")
        elif sim > 40 and skill_match > 30:
            just = (f"Candidate {cand_id} represents a strong potential fit. They display good alignment ({sim}%) "
                    f"with the role description and match skills: [{matched_str}]. However, they lack: [{missing_str}].")
        elif skill_match > 50:
            just = (f"Candidate {cand_id} possesses several core technical skills required [{matched_str}] "
                    f"({skill_match}% skill match), but their overall resume structure has a lower general textual overlap ({sim}%).")
        elif sim > 30:
            just = (f"Candidate {cand_id} has generic textual overlap ({sim}%), but lacks key required skills. "
                    f"Missing: [{missing_str}].")
        else:
            just = (f"Candidate {cand_id} shows weak overall alignment ({sim}% text similarity, {skill_match}% skill match). "
                    f"Missing critical requirements: [{missing_str}].")
        justifications.append(just)
        
    df_ranked["Fit_Justification"] = justifications
    
    # Sort: Highest Final Score -> Lowest Final Score
    df_ranked = df_ranked.sort_values(by="Final_Score", ascending=False).reset_index(drop=True)
    df_ranked["Rank"] = df_ranked.index + 1
    
    # Reorder columns for optimal presentation
    cols = ["Rank", "CAND_ID", "Category", "Similarity_Score", "Skill_Match_Score", "Final_Score", "Matched_Skills", "Missing_Skills", "Fit_Justification"]
    return df_ranked[cols]

def save_ranking_outputs(df_ranked, output_dir=r"D:\FUTURE_ML_O3\outputs"):
    """
    Saves the final ranking and skill gap analysis to separate CSV outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ranked Candidates Summary
    ranked_path = os.path.join(output_dir, "ranked_candidates.csv")
    df_ranked.to_csv(ranked_path, index=False)
    
    # 2. Complete Screening Results
    screening_path = os.path.join(output_dir, "screening_results.csv")
    df_ranked.to_csv(screening_path, index=False)
    
    # 3. Skill Gap Report
    gap_cols = ["CAND_ID", "Similarity_Score", "Skill_Match_Score", "Final_Score", "Matched_Skills", "Missing_Skills"]
    gap_report = df_ranked[gap_cols].rename(columns={
        "Matched_Skills": "Matched Skills",
        "Missing_Skills": "Missing Skills",
        "Skill_Match_Score": "Skill Match %"
    })
    gap_path = os.path.join(output_dir, "skill_gap_report.csv")
    gap_report.to_csv(gap_path, index=False)
    
    print(f"[CandidateRanking] Saved ranked_candidates.csv, screening_results.csv, and skill_gap_report.csv to: {output_dir}")

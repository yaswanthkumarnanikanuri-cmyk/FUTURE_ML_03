import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure standard professional visual styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.figsize": (10, 6)
})

def generate_eda_plots(df, extracted_skills_list, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
    """
    Generates and saves the exploratory data analysis plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("[Visualization] Generating EDA plots...")

    # Chart 1: Category Distribution
    plt.figure(figsize=(12, 6))
    cat_counts = df["Category"].value_counts().head(12)
    sns.barplot(x=cat_counts.values, y=cat_counts.index, hue=cat_counts.index, palette="viridis", legend=False)
    plt.title("Distribution of Resumes by Job Category (Top 12)")
    plt.xlabel("Number of Candidates")
    plt.ylabel("Job Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_distribution.png"), dpi=300)
    plt.close()

    # Chart 2: Top Skills in Resume Pool
    plt.figure(figsize=(10, 6))
    # Flatten list of skill sets
    all_skills = []
    for skill_set in extracted_skills_list:
        all_skills.extend(list(skill_set))
        
    skills_df = pd.Series(all_skills).value_counts().head(15)
    sns.barplot(x=skills_df.values, y=skills_df.index, hue=skills_df.index, palette="crest", legend=False)
    plt.title("Top 15 Most Common Technical Skills Across Candidate Pool")
    plt.xlabel("Frequency Count")
    plt.ylabel("Skill Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_skills.png"), dpi=300)
    plt.close()
    
    print("[Visualization] Saved category_distribution.png and top_skills.png.")

def generate_screening_plots(df_ranked, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
    """
    Generates and saves visual screening performance plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("[Visualization] Generating screening plots...")

    # Chart 3: Similarity Scores Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_ranked["Similarity_Score"], kde=True, color="#457b9d", bins=20)
    plt.axvline(df_ranked["Similarity_Score"].mean(), color="red", linestyle="--", label=f"Average ({df_ranked['Similarity_Score'].mean():.1f}%)")
    plt.title("Distribution of Resume-to-Job Textual Similarity Scores")
    plt.xlabel("Cosine Similarity Score (%)")
    plt.ylabel("Candidate Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_scores.png"), dpi=300)
    plt.close()

    # Chart 4: Candidate Ranking (Top 10)
    plt.figure(figsize=(10, 6))
    top_10 = df_ranked.head(10)
    sns.barplot(x="Final_Score", y="CAND_ID", data=top_10, hue="CAND_ID", palette="flare", legend=False)
    plt.xlim(0, 100)
    plt.title("Top 10 Ranked Candidates - Final Fit Score")
    plt.xlabel("Weighted Fit Score (%)")
    plt.ylabel("Candidate ID")
    # Annotate bar values
    for index, row in top_10.iterrows():
        plt.text(row["Final_Score"] + 1, index, f"{row['Final_Score']:.1f}%", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "candidate_ranking.png"), dpi=300)
    plt.close()

    # Chart 5: Skill Match Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_ranked["Skill_Match_Score"], kde=True, color="#2a9d8f", bins=15)
    plt.title("Distribution of Candidate Skill Match Scores")
    plt.xlabel("Required Skill Match Ratio (%)")
    plt.ylabel("Candidate Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "skill_match.png"), dpi=300)
    plt.close()

    # Chart 6: Skill Gap for Top 10 Candidates (Stacked comparison)
    plt.figure(figsize=(10, 6))
    # Parse count of matched and missing skills
    top_10_gap = top_10.copy()
    top_10_gap["Matched_Count"] = top_10_gap["Matched_Skills"].apply(lambda x: len(x.split(", ")) if x else 0)
    top_10_gap["Missing_Count"] = top_10_gap["Missing_Skills"].apply(lambda x: len(x.split(", ")) if x else 0)
    
    ind = np.arange(len(top_10_gap))
    width = 0.5
    
    p1 = plt.barh(ind, top_10_gap["Matched_Count"], width, color="#2a9d8f", label="Matched Skills")
    p2 = plt.barh(ind, top_10_gap["Missing_Count"], width, left=top_10_gap["Matched_Count"], color="#e76f51", label="Missing Skills")
    
    plt.title("Skill Gap Analysis: Matched vs. Missing Requirements (Top 10)")
    plt.yticks(ind, top_10_gap["CAND_ID"])
    plt.xlabel("Number of Technical Requirements")
    plt.ylabel("Candidate ID")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "skill_gap.png"), dpi=300)
    plt.close()

    print("[Visualization] Saved similarity_scores.png, candidate_ranking.png, skill_match.png, and skill_gap.png.")

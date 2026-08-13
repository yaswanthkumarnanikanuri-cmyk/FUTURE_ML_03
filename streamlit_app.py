import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pypdf
import io

from src.text_preprocessing import preprocess_text
from src.skill_extraction import extract_skills_from_text, SKILL_GROUPS
from src.job_parser import parse_job_description
from src.similarity_scoring import load_tfidf_vectorizer, calculate_cosine_similarity
from src.skill_gap_analysis import perform_skill_gap_analysis
from src.candidate_ranking import compute_final_scores_and_rank

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Subtitle
st.title("🤖 AI-Powered Resume Screening & Candidate Ranking")
st.markdown("---")

# Load preprocessed cache
@st.cache_data
def load_data():
    cache_path = os.path.join(os.path.dirname(__file__), "data", "preprocessed_resumes.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    else:
        st.error("Preprocessed dataset not found! Please run the pipeline script first.")
        return None

df_pool = load_data()

# Load preset job descriptions
@st.cache_data
def load_preset_jds():
    jd_path = os.path.join(os.path.dirname(__file__), "data", "job_descriptions.csv")
    if os.path.exists(jd_path):
        return pd.read_csv(jd_path)
    return None

jd_presets = load_preset_jds()

# Load pre-trained TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    return load_tfidf_vectorizer()

try:
    vectorizer = load_vectorizer()
except Exception as e:
    st.error(f"Error loading TF-IDF vectorizer: {e}. Please run the pipeline script first.")
    vectorizer = None

# Sidebar inputs
st.sidebar.header("🎯 Settings & Weighting")
sim_weight = st.sidebar.slider("Text Similarity Weight", 0.0, 1.0, 0.6, 0.05)
skill_weight = round(1.0 - sim_weight, 2)
st.sidebar.text(f"Skill Match Weight: {skill_weight}")

# Layout: Two Columns (Left = Inputs, Right = Results)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Job Description Details")
    
    # JD Input Option
    jd_option = st.radio("Choose Input Method:", ["Select Preset", "Paste Custom Job Description"])
    
    selected_jd_text = ""
    if jd_option == "Select Preset" and jd_presets is not None:
        selected_title = st.selectbox("Select Job Role Preset:", jd_presets["Job_Title"].tolist())
        selected_jd_text = jd_presets[jd_presets["Job_Title"] == selected_title].iloc[0]["Job_Description"]
        st.text_area("Preset Job Description Details:", value=selected_jd_text, height=180, disabled=True)
    else:
        selected_jd_text = st.text_area("Paste the Job Description here...", height=250)
        
    st.subheader("📂 Upload Custom Candidates (Optional)")
    uploaded_files = st.file_uploader("Upload candidate resumes (PDF format):", type=["pdf"], accept_multiple_files=True)

with col2:
    st.subheader("🧠 Screen & Rank Candidates")
    
    screen_button = st.button("Screen Candidates", type="primary")
    
    # Setup session state for screen results to persist
    if "df_ranked" not in st.session_state:
        st.session_state.df_ranked = None
        st.session_state.required_skills = None
        
    if screen_button:
        if not selected_jd_text.strip():
            st.warning("Please select or paste a job description first.")
        elif df_pool is None:
            st.error("Resume dataset pool is missing. Can't screen candidates.")
        else:
            with st.spinner("Processing resumes and matching requirements..."):
                # Parse job description
                parsed_jd = parse_job_description(selected_jd_text)
                required_skills = parsed_jd["required_skills"]
                st.session_state.required_skills = required_skills
                
                prep_jd = preprocess_text(parsed_jd["clean_text"])
                
                # Setup dataset
                current_pool = df_pool.copy()
                # Parse lists back from string
                import ast
                current_pool["Extracted_Skills"] = current_pool["Extracted_Skills"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                
                # Check for uploaded PDF files
                if uploaded_files:
                    uploaded_rows = []
                    for uploaded_file in uploaded_files:
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() or ""
                            
                        # Assign anonymous ID
                        cand_id = f"UPLOAD_{uploaded_file.name.split('.')[0]}"
                        prep_text = preprocess_text(text)
                        skills = extract_skills_from_text(text)
                        
                        uploaded_rows.append({
                            "CAND_ID": cand_id,
                            "Resume_str": text,
                            "Resume_html": "",
                            "Category": "Uploaded",
                            "Preprocessed_Resume": prep_text,
                            "Extracted_Skills": list(skills),
                            "Skill_Count": len(skills)
                        })
                    
                    df_uploaded = pd.DataFrame(uploaded_rows)
                    current_pool = pd.concat([df_uploaded, current_pool], ignore_index=True)
                
                # Calculate similarities
                similarities = calculate_cosine_similarity(vectorizer, current_pool["Preprocessed_Resume"], prep_jd)
                
                # Calculate skill matches
                skill_gap_results = []
                for idx, row in current_pool.iterrows():
                    cand_skills = set(row["Extracted_Skills"])
                    gap_res = perform_skill_gap_analysis(cand_skills, required_skills)
                    skill_gap_results.append(gap_res)
                    
                # Rank Candidates
                df_ranked = compute_final_scores_and_rank(current_pool, similarities, skill_gap_results, sim_weight, skill_weight)
                st.session_state.df_ranked = df_ranked
                st.success("Screening successfully completed!")

# Display Results
if st.session_state.df_ranked is not None:
    df_ranked = st.session_state.df_ranked
    required_skills = st.session_state.required_skills
    
    st.markdown("---")
    st.subheader("🏆 Candidate Leaderboard")
    
    # Format table for presentation
    display_cols = ["Rank", "CAND_ID", "Category", "Similarity_Score", "Skill_Match_Score", "Final_Score"]
    st.dataframe(
        df_ranked[display_cols].rename(columns={
            "CAND_ID": "Candidate ID",
            "Similarity_Score": "Text Similarity (%)",
            "Skill_Match_Score": "Skill Match (%)",
            "Final_Score": "Overall Fit Score (%)"
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Candidate details viewer
    st.subheader("🔍 Individual Candidate Analysis")
    cand_options = df_ranked["CAND_ID"].tolist()
    selected_cand = st.selectbox("Select Candidate to Inspect Details:", cand_options)
    
    if selected_cand:
        cand_row = df_ranked[df_ranked["CAND_ID"] == selected_cand].iloc[0]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Overall Fit Score", f"{cand_row['Final_Score']}%")
        with c2:
            st.metric("Textual Similarity", f"{cand_row['Similarity_Score']}%")
        with c3:
            st.metric("Skill Match Percentage", f"{cand_row['Skill_Match_Score']}%")
            
        st.markdown(f"**Fit Justification:** *{cand_row['Fit_Justification']}*")
        
        # Display matched and missing skills
        c4, c5 = st.columns(2)
        with c4:
            st.success("Matched Required Skills")
            matched_list = cand_row["Matched_Skills"].split(", ") if cand_row["Matched_Skills"] else []
            if matched_list:
                for s in matched_list:
                    st.markdown(f"✔️ {s}")
            else:
                st.info("No required skills matched.")
        with c5:
            st.error("Missing Required Skills")
            missing_list = cand_row["Missing_Skills"].split(", ") if cand_row["Missing_Skills"] else []
            if missing_list:
                for s in missing_list:
                    st.markdown(f"❌ {s}")
            else:
                st.info("No required skills missing.")

    st.markdown("---")
    st.subheader("📊 Screening Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Fit Score Distribution", "Similarity vs Skill Match", "Skill Gap Chart"])
    
    with tab1:
        # Chart 1: Fit score distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df_ranked["Final_Score"], kde=True, color="#2b6cb0", ax=ax, bins=15)
        ax.set_title("Distribution of Overall Fit Scores")
        ax.set_xlabel("Fit Score (%)")
        ax.set_ylabel("Frequency Count")
        st.pyplot(fig)
        
    with tab2:
        # Chart 2: Scatter plot of Similarity vs Skill Match
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=df_ranked, 
            x="Similarity_Score", 
            y="Skill_Match_Score", 
            hue="Category", 
            size="Final_Score",
            sizes=(50, 300),
            ax=ax,
            palette="Set2"
        )
        ax.set_title("Candidate Mapping: Similarity vs. Skill Match Score")
        ax.set_xlabel("Text Similarity Score (%)")
        ax.set_ylabel("Skill Match Score (%)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        st.pyplot(fig)
        
    with tab3:
        # Chart 3: Stacked Bar Plot of Matched vs Missing for top 10
        fig, ax = plt.subplots(figsize=(10, 5))
        top_10 = df_ranked.head(10).copy()
        top_10["Matched_Count"] = top_10["Matched_Skills"].apply(lambda x: len(x.split(", ")) if x else 0)
        top_10["Missing_Count"] = top_10["Missing_Skills"].apply(lambda x: len(x.split(", ")) if x else 0)
        
        ind = np.arange(len(top_10))
        width = 0.5
        
        ax.barh(ind, top_10["Matched_Count"], width, color="#2a9d8f", label="Matched Skills")
        ax.barh(ind, top_10["Missing_Count"], width, left=top_10["Matched_Count"], color="#e76f51", label="Missing Skills")
        
        ax.set_title("Skill Alignment Analysis (Top 10 Candidates)")
        ax.set_yticks(ind)
        ax.set_yticklabels(top_10["CAND_ID"])
        ax.set_xlabel("Number of Requirements")
        ax.legend()
        st.pyplot(fig)

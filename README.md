# 🤖 AI-Powered Resume Screening & Candidate Ranking System

An end-to-end Natural Language Processing (NLP) and Machine Learning decision-support system designed to automate, visualize, and optimize candidate screening for corporate recruiters.

---

## 🎯 Why I Built This
Reviewing hundreds of applicant resumes for technical roles is a slow, manually intensive, and fatigue-prone task. It often leads to recruitment bottlenecks and introduces inconsistent evaluation criteria. 

I built this system to demonstrate a clean, transparent, and explainable NLP approach to resume screening. Rather than acting as an autonomous hiring tool, it serves as a **recruiter decision-support system** that programmatically extracts skills, matches applicant profiles, highlights critical qualification gaps, and ranks candidates to reduce initial screening times from days to minutes.

---

## 💡 What This Project Does
The system implements a structured text engineering and similarity matching pipeline:

```
[Job Description] ──> [Skill Parser] ──────────────────────┐
                                                           ▼
[Candidate Resumes] ──> [Text Preprocessing] ──> [Skill Gap Analysis] ──> [Hybrid Scoring] ──> [Ranked List]
                             │                                                 ▲
                             └──> [TF-IDF Vectorizer] ──> [Cosine Similarity] ┘
```

1. **Job Description Parsing**: Accepts a job description and automatically extracts required technical competencies.
2. **Text Preprocessing**: Normalizes, cleans (removes HTML tags, URLs, metadata), and lemmatizes raw resume and job text.
3. **Skill Dictionary Match**: Evaluates candidates against a configurable regular expression dictionary mapping 40+ industry concepts.
4. **TF-IDF & Cosine Similarity**: Vectorizes resumes to extract semantic text overlap percentages.
5. **Skill Gap Computations**: Cross-checks qualifications to identify matched and missing competencies.
6. **Leaderboard Ranking**: Combines matching scores to rank candidates.
7. **Interactive Dashboard**: Launches an interactive Streamlit UI.

---

## ✨ Key Features
- **Technical Tag Preservation**: Features a tokenizer regex context that preserves tags like `C++`, `C#`, and `.NET` which are destroyed by standard tokenizers.
- **Configurable Fit Scoring**: Let recruiters tune similarity and skill match weighting in real time.
- **Explainable Match Justification**: Generates a contextual textual summary explaining *why* a candidate received their fit score.
- **Visual Analytics**: Interactive data analytics (category distributions, skill gap bar charts, scatter plots).
- **Custom PDF Uploads**: Extracted text parser using `pypdf` to screen local applicant files.

---

## 🧠 How the System Thinks
The screening system relies on three distinct layers to rank candidates:
- **TF-IDF Vectorization**: Converts preprocessed documents into numerical matrices weighting terms relative to their corpus-wide frequency.
- **Cosine Similarity**: Calculates the angle between candidate resume vectors and the job description vector to yield a text similarity percentage.
- **Skill Match Score**: An exact lexical scan of mandatory keywords to determine the percentage of required skills a candidate possesses.
- **Fit Scoring**: Aggregates lexical matches and semantic similarities into a final weighted score.

---

## 📊 Scoring Method
The system calculates applicant fit using a configurable, transparent scoring formula:

$$\text{Final Score} = (0.60 \times \text{Text Similarity \\%}) + (0.40 \times \text{Skill Match \\%})$$

This formulation ensures that a candidate who has good general textual overlap but lacks mandatory technical skills is not ranked above an applicant who fits the core technical requirements.

---

## 🗂️ Dataset
The system utilizes the **Resume Dataset — Kaggle** by Sneha Anbhaval:
- **Source**: [https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Content**: 2,400+ resumes scraped from livecareer.com across 24 job categories.
- **Columns**: `ID`, `Resume_str`, `Resume_html`, and `Category`.
- **Pre-populated Job Descriptions**: Includes simulated requirements for common roles (e.g. Data Scientist, Machine Learning Engineer, Data Analyst) stored in `data/job_descriptions.csv`.

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core execution logic and script modules |
| **Pandas / NumPy** | Tabular data structures and matrix operations |
| **NLTK / spaCy** | Lemmatization, clean stopword filtering, and NLP preprocessing |
| **Scikit-learn** | TF-IDF vocabulary modeling and Cosine Similarity calculation |
| **Streamlit** | Multi-page interactive recruiter dashboard |
| **ReportLab** | Programmatic PDF report creation |
| **PyPDF** | Custom uploaded PDF resume text extraction |
| **Matplotlib / Seaborn** | Analytical chart generation |

---

## 🏗️ Project Architecture
```
D:\FUTURE_ML_O3\
├── data/
│   ├── Resume.csv               # Extracted Kaggle Resume Dataset
│   └── job_descriptions.csv     # Simulated job description presets
├── notebook/
│   └── Resume_Candidate_Screening.ipynb # Interactive pipeline notebook
├── src/
│   ├── data_loader.py           # Dataset loaders and cleaning
│   ├── text_preprocessing.py    # NLP preprocessors and text cleaners
│   ├── skill_extraction.py      # Technical skill dictionaries and matchers
│   ├── job_parser.py            # JD skill requirements parsers
│   ├── similarity_scoring.py    # TF-IDF representations and cosine similarities
│   ├── candidate_ranking.py     # Aggregators, scoring, and rank lists
│   ├── skill_gap_analysis.py    # Matched vs missing skill categorizers
│   └── visualization.py         # Matplotlib and Seaborn analytical plots
├── models/
│   └── tfidf_vectorizer.pkl     # Serialized TF-IDF model
├── outputs/
│   ├── ranked_candidates.csv    # Final candidate leaderboard list
│   ├── screening_results.csv    # Full candidates metadata logs
│   └── skill_gap_report.csv     # Candidate skill gap logs
├── images/
│   ├── category_distribution.png
│   ├── top_skills.png
│   ├── similarity_scores.png
│   ├── candidate_ranking.png
│   ├── skill_match.png
│   └── skill_gap.png
├── report/
│   └── FUTURE_ML_O3_Project_Report.pdf # Formatted PDF Project Report
├── app.py                       # Streamlit Application Dashboard
├── requirements.txt             # Dependency declarations
├── .gitignore                   # Git ignore files
└── data_downloader.py           # Pipeline setup dataset downloader
```

---

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/FUTURE_ML_O3.git
   cd FUTURE_ML_O3
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Extract Dataset**
   Run the downloader script to programmatically retrieve, unpack, and place the Kaggle dataset into the `data/` folder:
   ```bash
   python data_downloader.py
   ```

---

## 💻 How to Run

### Run the Pipeline & Train the Model
Run the pipeline to process resumes, train the TF-IDF vectorizer, compute candidate fit scores against a baseline MLE job description, and export visualizations:
```bash
python run_pipeline.py
```

### Launch the Streamlit Dashboard
Run the web application to screen custom JDs and upload your own resumes:
```bash
streamlit run app.py
```

### View Project Notebook
Open the Jupyter notebook to review step-by-step code details:
```bash
jupyter notebook notebook/Resume_Candidate_Screening.ipynb
```

---

## 📈 Pipeline Results & Analytics
Running the baseline pipeline for the **Machine Learning Engineer** job description yielded the following outcomes:
- **Corpus deduplication**: Removed duplicate resumes, resulting in 2,482 unique candidate files.
- **Baseline JD requirements detected**: `['Machine Learning', 'SQL', 'AWS', 'Docker', 'Scikit-learn', 'Python', 'NLP', 'Git', 'Pandas', 'NumPy']`.
- **Top 5 candidates leaderboard ranking**:
  1. `CAND_906` - Category: Business-Development | Similarity: 43.1% | Skill Match: 70.0% | Fit Score: 53.86%
  2. `CAND_364` - Category: Chef | Similarity: 51.5% | Skill Match: 50.0% | Fit Score: 50.90%
  3. `CAND_449` - Category: Chef | Similarity: 44.9% | Skill Match: 50.0% | Fit Score: 46.94%
  4. `CAND_1650` - Category: Engineering | Similarity: 44.5% | Skill Match: 50.0% | Fit Score: 46.70%
  5. `CAND_1847` - Category: Database | Similarity: 52.3% | Skill Match: 30.0% | Fit Score: 43.38%

---

## 📸 Screenshots & Visualizations
Generated plots are saved inside the [images/](file:///D:/FUTURE_ML_O3/images) folder:
* **Category Distribution**: `images/category_distribution.png`
* **Top Skills**: `images/top_skills.png`
* **Similarity Score Distribution**: `images/similarity_scores.png`
* **Leaderboard Scores**: `images/candidate_ranking.png`
* **Skill Match Distribution**: `images/skill_match.png`
* **Skill Gap Stacked Charts**: `images/skill_gap.png`

---

## 🔎 Example Workflow

### 1. Input Job Description (Machine Learning Engineer)
> *"We are looking for a Machine Learning Engineer with Python, SQL, Pandas, NumPy, Scikit-learn, Machine Learning, and NLP experience. Experience with AWS, Docker, and Git is preferred."*

### 2. Candidate Screening Report (e.g. CAND_906)
- **Textual Similarity**: `43.1%`
- **Skill Match Score**: `70.0%`
- **Overall Fit Score**: `53.86%`
- **Matched Skills**: `AWS, Docker, Git, Machine Learning, Pandas, Python, SQL`
- **Missing Skills**: `NLP, NumPy, Scikit-learn`
- **Recruiter Justification**: 
  *"Candidate CAND_906 represents a strong potential fit. They display good alignment (43.1%) with the role description and match skills: [AWS, Docker, Git, Machine Learning, Pandas, Python, SQL]. However, they lack: [NLP, NumPy, Scikit-learn]."*

---

## 💼 Business Value
- **Reduces Screening Bottlenecks**: Allows recruiting teams to filter down a pool of thousands of resumes to a top-tier shortlist in seconds.
- **Uncovers Hidden Talent**: Highlights candidates who possess correct technical skills but might have a low textual similarity due to non-traditional phrasing.
- **Visual HR Analytics**: Provides demographic maps of skills and categorical representation.

---

## ⚠️ Limitations & Ethical Considerations
- **Decision-Support Only**: This prototype is a screening aid and should never be used to make final automated hiring decisions. Human-in-the-loop review is mandatory.
- **Exact Keyword matching limitations**: The keyword skill dictionary uses regex, which matches terms literally. It does not evaluate semantic depth or the context of how a skill was applied.
- **Formatting Biases**: Candidates who write long resumes containing multiple repetitions of skill keywords will artificially score higher on TF-IDF similarity.
- **Anonymization**: Resumes are processed anonymously (CAND_ID) to minimize potential gender, ethnic, or name-based biases during the initial ranking.

---

## 🚀 Future Improvements
- **Semantic Vector Embeddings**: Transition from TF-IDF to Sentence-BERT or OpenAI text embeddings to capture synonyms and semantic meanings.
- **Named Entity Recognition (NER)**: Train a custom spaCy NER model to identify and isolate candidate certifications, education, and years of experience.
- **Automatic Bias Audits**: Integrate fairness toolkits (e.g. Fairlearn) to flag and balance category distribution.

---

## 👨💻 Internship Details
- **Project Purpose**: Future Interns Machine Learning Internship — Task 3
- **Objective**: Develop an NLP and ML Candidate Screening & Skill Gap Dashboard.

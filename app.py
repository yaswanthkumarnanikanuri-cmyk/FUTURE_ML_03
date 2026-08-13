import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ast
import io
import base64
import pypdf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template_string, jsonify

from src.text_preprocessing import preprocess_text
from src.skill_extraction import extract_skills_from_text
from src.job_parser import parse_job_description
from src.similarity_scoring import load_tfidf_vectorizer, calculate_cosine_similarity
from src.skill_gap_analysis import perform_skill_gap_analysis
from src.candidate_ranking import compute_final_scores_and_rank

# Initialize Flask application
app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "data", "preprocessed_resumes.csv")
JD_PATH = os.path.join(BASE_DIR, "data", "job_descriptions.csv")

# Load data helper
def get_data_pool():
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    return None

# Load presets helper
def get_preset_jds():
    if os.path.exists(JD_PATH):
        return pd.read_csv(JD_PATH)
    return None

# Load TF-IDF vectorizer
try:
    vectorizer = load_tfidf_vectorizer()
except Exception:
    vectorizer = None

# Global presets
preset_jds = get_preset_jds()
df_pool_raw = get_data_pool()

# Custom inline HTML page for premium UI aesthetics
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Resume Screening Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f7fafc;
            --card-bg: #ffffff;
            --text-main: #2d3748;
            --text-muted: #718096;
            --primary: #2b6cb0;
            --primary-hover: #2b5c8f;
            --accent: #319795;
            --border: #e2e8f0;
            --success: #2f855a;
            --error: #9b2c2c;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
        }

        header {
            background-color: var(--card-bg);
            border-bottom: 1px solid var(--border);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        header h1 {
            font-size: 24px;
            margin: 0;
            color: var(--primary);
        }

        .container {
            max-width: 1400px;
            margin: 40px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 30px;
        }

        h2 {
            font-size: 20px;
            margin-top: 0;
            border-bottom: 2px solid var(--border);
            padding-bottom: 10px;
            color: var(--primary);
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-weight: 500;
            margin-bottom: 8px;
        }

        textarea {
            width: 100%;
            height: 180px;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            resize: vertical;
            box-sizing: border-box;
        }

        select, input[type="text"], input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            box-sizing: border-box;
        }

        .slider-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
        }

        .slider-container input {
            flex-grow: 1;
        }

        button {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
            transition: background-color 0.2s;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        .table-container {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        th, td {
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }

        th {
            background-color: #f7fafc;
            font-weight: 600;
        }

        .badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            background-color: #edf2f7;
        }

        .badge-fit {
            background-color: #ebf8ff;
            color: #2b6cb0;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .metric-card {
            background-color: #f7fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }

        .metric-title {
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 20px;
            font-weight: 700;
            color: var(--primary);
        }

        .skills-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }

        .skill-box {
            padding: 15px;
            border-radius: 8px;
        }

        .skill-box-matched {
            background-color: #f0fff4;
            border: 1px solid #c6f6d5;
            color: var(--success);
        }

        .skill-box-missing {
            background-color: #fff5f5;
            border: 1px solid #fed7d7;
            color: var(--error);
        }

        .skill-list {
            margin: 0;
            padding-left: 20px;
        }

        .chart-img {
            width: 100%;
            border-radius: 8px;
            margin-top: 15px;
            border: 1px solid var(--border);
        }

        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
        }

        .tab-btn {
            background: none;
            color: var(--text-muted);
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-weight: 500;
            width: auto;
            border-bottom: 2px solid transparent;
        }

        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <header>
        <h1>🤖 AI Resume Screening System</h1>
        <span class="badge badge-fit">Task 3 Candidate Scorer</span>
    </header>

    <div class="container">
        <!-- Input Panel -->
        <div>
            <form action="/screen" method="POST" enctype="multipart/form-data" class="card">
                <h2>📋 Job Requirements</h2>
                
                <div class="form-group">
                    <label>Choose Input Method:</label>
                    <select id="jd_type" name="jd_type" onchange="toggleJDInput()">
                        <option value="preset" {% if selected_type == 'preset' %}selected{% endif %}>Select Preset Role</option>
                        <option value="custom" {% if selected_type == 'custom' %}selected{% endif %}>Paste Custom Description</option>
                    </select>
                </div>

                <div class="form-group" id="preset_group">
                    <label>Select Job Role Preset:</label>
                    <select name="preset_title" onchange="this.form.submit()">
                        {% for preset in presets %}
                        <option value="{{ preset }}" {% if preset == selected_preset %}selected{% endif %}>{{ preset }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label>Job Description Details:</label>
                    <textarea name="jd_text" id="jd_text" required>{{ jd_text }}</textarea>
                </div>

                <div class="form-group">
                    <label>Weight Configuration:</label>
                    <div class="slider-container">
                        <span>Cosine Similarity:</span>
                        <input type="range" name="sim_weight" min="0.0" max="1.0" step="0.05" value="{{ sim_weight }}" oninput="updateWeights(this.value)">
                        <span id="sim_lbl">{{ sim_weight }}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--text-muted); margin-top: 5px;">
                        Remaining weight assigned to exact skill matches: <span id="skill_lbl">{{ skill_weight }}</span>
                    </div>
                </div>

                <div class="form-group">
                    <label>Upload Custom Resumes (PDF):</label>
                    <input type="file" name="resumes" multiple accept=".pdf">
                </div>

                <button type="submit">Screen Candidates</button>
            </form>
        </div>

        <!-- Output Panel -->
        <div>
            {% if ranked_candidates is not None %}
            <div class="card">
                <h2>🏆 Candidate Leaderboard</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Candidate ID</th>
                                <th>Category</th>
                                <th>Similarity (%)</th>
                                <th>Skill Match (%)</th>
                                <th>Final Score (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in ranked_candidates %}
                            <tr style="cursor:pointer;" onclick="viewCandidate('{{ row.CAND_ID }}')">
                                <td><b>{{ row.Rank }}</b></td>
                                <td>{{ row.CAND_ID }}</td>
                                <td><span class="badge">{{ row.Category }}</span></td>
                                <td>{{ row.Similarity_Score }}%</td>
                                <td>{{ row.Skill_Match_Score }}%</td>
                                <td><b>{{ row.Final_Score }}%</b></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Details Section -->
            <div class="card" id="details_section">
                <h2>🔍 Candidate Detail Analysis</h2>
                <div class="form-group">
                    <label>Select Candidate to Inspect Details:</label>
                    <select id="cand_select" onchange="showCandDetails(this.value)">
                        {% for row in ranked_candidates %}
                        <option value="{{ row.CAND_ID }}">{{ row.CAND_ID }} ({{ row.Category }})</option>
                        {% endfor %}
                    </select>
                </div>

                {% for row in ranked_candidates %}
                <div class="cand-details-div" id="details_{{ row.CAND_ID }}" style="display: {% if loop.first %}block{% else %}none{% endif %};">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-title">Overall Fit</div>
                            <div class="metric-value">{{ row.Final_Score }}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Similarity</div>
                            <div class="metric-value">{{ row.Similarity_Score }}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Skill Match</div>
                            <div class="metric-value">{{ row.Skill_Match_Score }}%</div>
                        </div>
                    </div>
                    <p style="margin-top: 15px; line-height: 1.5;">
                        <b>Justification:</b> <i>{{ row.Fit_Justification }}</i>
                    </p>

                    <div class="skills-grid">
                        <div class="skill-box skill-box-matched">
                            <strong>Matched Skills:</strong>
                            <ul class="skill-list">
                                {% if row.Matched_Skills %}
                                    {% for skill in row.Matched_Skills.split(', ') %}
                                        <li>{{ skill }}</li>
                                    {% endfor %}
                                {% else %}
                                    <li>None</li>
                                {% endif %}
                            </ul>
                        </div>
                        <div class="skill-box skill-box-missing">
                            <strong>Missing Skills:</strong>
                            <ul class="skill-list">
                                {% if row.Missing_Skills %}
                                    {% for skill in row.Missing_Skills.split(', ') %}
                                        <li>{{ skill }}</li>
                                    {% endfor %}
                                {% else %}
                                    <li>None</li>
                                {% endif %}
                            </ul>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- Charts Section -->
            <div class="card">
                <h2>📊 Screening Analytics Dashboard</h2>
                <div class="tabs">
                    <button type="button" class="tab-btn active" onclick="switchTab(event, 'tab_dist')">Fit Distribution</button>
                    <button type="button" class="tab-btn" onclick="switchTab(event, 'tab_scatter')">Similarity vs Match</button>
                    <button type="button" class="tab-btn" onclick="switchTab(event, 'tab_gap')">Skill Gap Chart</button>
                </div>

                <div id="tab_dist" class="tab-content active">
                    <img class="chart-img" src="data:image/png;base64,{{ chart1 }}" alt="Fit Score Distribution">
                </div>
                <div id="tab_scatter" class="tab-content">
                    <img class="chart-img" src="data:image/png;base64,{{ chart2 }}" alt="Similarity vs Skill Match">
                </div>
                <div id="tab_gap" class="tab-content">
                    <img class="chart-img" src="data:image/png;base64,{{ chart3 }}" alt="Skill Alignment Chart">
                </div>
            </div>
            {% else %}
            <div class="card" style="text-align: center; padding: 50px;">
                <p style="font-size: 18px; color: var(--text-muted);">Please click "Screen Candidates" to analyze the pool.</p>
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        function toggleJDInput() {
            var type = document.getElementById("jd_type").value;
            var presetGroup = document.getElementById("preset_group");
            if (type === "preset") {
                presetGroup.style.display = "block";
            } else {
                presetGroup.style.display = "none";
            }
        }

        function updateWeights(val) {
            document.getElementById("sim_lbl").innerText = val;
            document.getElementById("skill_lbl").innerText = (1.0 - parseFloat(val)).toFixed(2);
        }

        function showCandDetails(candId) {
            var divs = document.getElementsByClassName("cand-details-div");
            for (var i = 0; i < divs.length; i++) {
                divs[i].style.display = "none";
            }
            document.getElementById("details_" + candId).style.display = "block";
            document.getElementById("cand_select").value = candId;
        }

        function viewCandidate(candId) {
            var select = document.getElementById("cand_select");
            select.value = candId;
            showCandDetails(candId);
            document.getElementById("details_section").scrollIntoView({ behavior: 'smooth' });
        }

        function switchTab(evt, tabId) {
            var contents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < contents.length; i++) {
                contents[i].style.display = "none";
                contents[i].classList.remove("active");
            }
            var buttons = document.getElementsByClassName("tab-btn");
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove("active");
            }
            document.getElementById(tabId).style.display = "block";
            document.getElementById(tabId).classList.add("active");
            evt.currentTarget.classList.add("active");
        }

        // Initialize display
        toggleJDInput();
    </script>
</body>
</html>
"""

# Default weights
DEFAULT_SIM_WEIGHT = 0.60
DEFAULT_SKILL_WEIGHT = 0.40

# App Routes
@app.route("/", methods=["GET"])
def index():
    presets = []
    default_jd_text = ""
    selected_preset = ""
    
    if preset_jds is not None:
        presets = preset_jds["Job_Title"].tolist()
        if presets:
            selected_preset = presets[0]
            default_jd_text = preset_jds.iloc[0]["Job_Description"]

    return render_template_string(
        HTML_TEMPLATE,
        presets=presets,
        selected_preset=selected_preset,
        jd_text=default_jd_text,
        sim_weight=DEFAULT_SIM_WEIGHT,
        skill_weight=DEFAULT_SKILL_WEIGHT,
        selected_type="preset",
        ranked_candidates=None
    )

@app.route("/screen", methods=["POST"])
def screen():
    presets = []
    if preset_jds is not None:
        presets = preset_jds["Job_Title"].tolist()

    # Form parameters
    jd_type = request.form.get("jd_type", "preset")
    sim_weight = float(request.form.get("sim_weight", DEFAULT_SIM_WEIGHT))
    skill_weight = round(1.0 - sim_weight, 2)
    selected_preset = request.form.get("preset_title", "")
    
    # Text determination
    if jd_type == "preset" and preset_jds is not None and selected_preset:
        jd_text = preset_jds[preset_jds["Job_Title"] == selected_preset].iloc[0]["Job_Description"]
    else:
        jd_text = request.form.get("jd_text", "")

    # Parsing JD
    parsed_jd = parse_job_description(jd_text)
    required_skills = parsed_jd["required_skills"]
    prep_jd = preprocess_text(parsed_jd["clean_text"])

    # Load dataset
    current_pool = df_pool_raw.copy()
    current_pool["Extracted_Skills"] = current_pool["Extracted_Skills"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Process uploaded PDF files
    uploaded_files = request.files.getlist("resumes")
    uploaded_rows = []
    
    for u_file in uploaded_files:
        if u_file and u_file.filename.endswith(".pdf"):
            try:
                pdf_reader = pypdf.PdfReader(u_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                
                # Preprocess uploaded file
                cand_id = f"UPLOAD_{u_file.filename.split('.')[0]}"
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
            except Exception as e:
                print(f"Failed to parse uploaded PDF: {e}")

    if uploaded_rows:
        df_uploaded = pd.DataFrame(uploaded_rows)
        current_pool = pd.concat([df_uploaded, current_pool], ignore_index=True)

    # Run predictions
    similarities = calculate_cosine_similarity(vectorizer, current_pool["Preprocessed_Resume"], prep_jd)
    
    skill_gap_results = []
    for idx, row in current_pool.iterrows():
        cand_skills = set(row["Extracted_Skills"])
        gap_res = perform_skill_gap_analysis(cand_skills, required_skills)
        skill_gap_results.append(gap_res)

    df_ranked = compute_final_scores_and_rank(current_pool, similarities, skill_gap_results, sim_weight, skill_weight)
    
    # Limit to top 25 for fast response rendering
    df_ranked_limit = df_ranked.head(25)
    candidates_list = df_ranked_limit.to_dict(orient="records")

    # Generate charts as base64 in-memory bytes to avoid disk writes
    # Plot 1: Fit score distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df_ranked["Final_Score"], kde=True, color="#2b6cb0", ax=ax, bins=12)
    ax.set_title("Overall Fit Score Distribution")
    ax.set_xlabel("Fit Score (%)")
    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=150)
    plt.close()
    buf1.seek(0)
    chart1 = base64.b64encode(buf1.getvalue()).decode('utf-8')

    # Plot 2: Scatter plot of Similarity vs Skill Match
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.scatterplot(
        data=df_ranked_limit, 
        x="Similarity_Score", 
        y="Skill_Match_Score", 
        hue="Category", 
        size="Final_Score",
        sizes=(20, 150),
        ax=ax,
        palette="Set2"
    )
    ax.set_title("Similarity vs. Skill Match Score")
    ax.set_xlabel("Similarity (%)")
    ax.set_ylabel("Skill Match (%)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=150)
    plt.close()
    buf2.seek(0)
    chart2 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    # Plot 3: Stacked Bar Plot of Matched vs Missing for top 10
    fig, ax = plt.subplots(figsize=(6, 3))
    top_10 = df_ranked_limit.head(10).copy()
    top_10["Matched_Count"] = top_10["Matched_Skills"].apply(lambda x: len(x.split(", ")) if x else 0)
    top_10["Missing_Count"] = top_10["Missing_Skills"].apply(lambda x: len(x.split(", ")) if x else 0)
    
    ind = np.arange(len(top_10))
    width = 0.4
    
    ax.barh(ind, top_10["Matched_Count"], width, color="#2a9d8f", label="Matched")
    ax.barh(ind, top_10["Missing_Count"], width, left=top_10["Matched_Count"], color="#e76f51", label="Missing")
    ax.set_title("Skill Alignment (Top 10)")
    ax.set_yticks(ind)
    ax.set_yticklabels(top_10["CAND_ID"], fontsize='small')
    ax.set_xlabel("Count")
    ax.legend(fontsize='small')
    buf3 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf3, format='png', dpi=150)
    plt.close()
    buf3.seek(0)
    chart3 = base64.b64encode(buf3.getvalue()).decode('utf-8')

    return render_template_string(
        HTML_TEMPLATE,
        presets=presets,
        selected_preset=selected_preset,
        jd_text=jd_text,
        sim_weight=sim_weight,
        skill_weight=skill_weight,
        selected_type=jd_type,
        ranked_candidates=candidates_list,
        chart1=chart1,
        chart2=chart2,
        chart3=chart3
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

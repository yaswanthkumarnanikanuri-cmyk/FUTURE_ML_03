import re

# Configurable dictionary of skills mapped to their matching regex patterns.
# The keys represent the standard display names of the skills.
# The values are regular expressions designed to capture various abbreviations and aliases.
SKILL_PATTERNS = {
    # Programming Languages
    "Python": r"\bpython\b",
    "Java": r"\bjava\b(?!script)",
    "C": r"\bc\b",
    "C++": r"\bc\+\+\b",
    "C#": r"\bc\#\b",
    "JavaScript": r"\bjavascript\b|\bjs\b",
    "HTML": r"\bhtml5?\b",
    "CSS": r"\bcss3?\b",
    "SQL": r"\bsql\b",
    
    # Databases
    "MySQL": r"\bmysql\b",
    "PostgreSQL": r"\bpostgresql\b|\bpostgres\b",
    "MongoDB": r"\bmongodb\b|\bmongo\b",
    
    # Data Analysis & BI
    "Excel": r"\bexcel\b",
    "Power BI": r"\bpower\s?bi\b",
    "Tableau": r"\btableau\b",
    
    # Data Science Libraries
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "Matplotlib": r"\bmatplotlib\b",
    "Seaborn": r"\bseaborn\b",
    "Scikit-learn": r"\bscikit[-_\s]?learn\b|\bsklearn\b",
    
    # Deep Learning Frameworks
    "TensorFlow": r"\btensorflow\b|\btf\b",
    "PyTorch": r"\bpytorch\b",
    "Keras": r"\bkeras\b",
    
    # Core AI / ML
    "Machine Learning": r"\bmachine[-_\s]?learning\b|\bml\b",
    "Deep Learning": r"\bdeep[-_\s]?learning\b|\bdl\b",
    "NLP": r"\bnlp\b|\bnatural\s+language\s+processing\b",
    "Computer Vision": r"\bcomputer\s+vision\b|\bcv\b",
    "Generative AI": r"\bgenerative\s+ai\b|\bgen\s?ai\b",
    "LLM": r"\bllm\b|\blarge\s+language\s+models?\b",
    "RAG": r"\brag\b|\bretrieval\s+augmented\s+generation\b",
    "LangChain": r"\blangchain\b",
    
    # Cloud & DevOps
    "AWS": r"\baws\b|\bamazon\s+web\s+services\b",
    "Azure": r"\bazure\b",
    "GCP": r"\bgcp\b|\bgoogle\s+cloud\s+(platform\s+)?\b",
    "Docker": r"\bdocker\b",
    "Git": r"\bgit\b|\bgithub\b|\bgitlab\b",
    "Linux": r"\blinux\b|\bunix\b",
    
    # Web Frameworks
    "Django": r"\bdjango\b",
    "Flask": r"\bflask\b",
    "FastAPI": r"\bfastapi\b",
    
    # Big Data
    "Spark": r"\bspark\b|\bpyspark\b|\bapache\s+spark\b",
    "Hadoop": r"\bhadoop\b",
    
    # Concepts
    "Statistics": r"\bstatistics?\b|\bstatistical\b",
    "Data Analysis": r"\bdata\s+analysis\b|\bdata\s+analytics\b",
    "Data Science": r"\bdata\s+science\b",
}

# Group mapping for structured visualization/reporting
SKILL_GROUPS = {
    "Programming": ["Python", "Java", "C", "C++", "C#", "JavaScript", "HTML", "CSS", "SQL"],
    "Databases": ["MySQL", "PostgreSQL", "MongoDB"],
    "BI & Analytics": ["Excel", "Power BI", "Tableau", "Data Analysis", "Statistics"],
    "ML/DS Libraries": ["Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn"],
    "Deep Learning": ["TensorFlow", "PyTorch", "Keras"],
    "Advanced AI": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Generative AI", "LLM", "RAG", "LangChain", "Data Science"],
    "Cloud & DevOps": ["AWS", "Azure", "GCP", "Docker", "Git", "Linux"],
    "Web Development": ["Django", "Flask", "FastAPI"],
    "Big Data": ["Spark", "Hadoop"]
}

def extract_skills_from_text(text):
    """
    Scans a given text (resume or job description) and extracts matched skills.
    Returns a set of standard display names of the matched skills.
    """
    if not isinstance(text, str):
        return set()
        
    extracted_skills = set()
    # Normalize input for regex matching (preserve spacing around symbols like +, #)
    norm_text = text.lower()
    norm_text = re.sub(r'(?<=[a-zA-Z])\+', ' +', norm_text)
    norm_text = re.sub(r'\+(?=[a-zA-Z])', '+ ', norm_text)
    norm_text = re.sub(r'\s+', ' ', norm_text)
    
    for skill_name, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, norm_text):
            extracted_skills.add(skill_name)
            
    return extracted_skills

def get_skill_category(skill_name):
    """
    Determines the group category for a given skill.
    """
    for category, skills in SKILL_GROUPS.items():
        if skill_name in skills:
            return category
    return "Other"

if __name__ == "__main__":
    sample_cv = "Experienced Python developer with a strong background in Machine Learning, Scikit-learn, Pandas, NumPy and SQL. Familiar with docker and aws."
    skills = extract_skills_from_text(sample_cv)
    print("Sample CV:", sample_cv)
    print("Extracted Skills:", list(skills))

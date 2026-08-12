from src.skill_extraction import extract_skills_from_text

def parse_job_description(jd_text):
    """
    Parses a job description to extract required technical skills.
    
    Parameters:
    jd_text (str): Raw text of the job description.
    
    Returns:
    dict: A dictionary containing extracted 'required_skills' (set) and 'clean_text' (str).
    """
    if not isinstance(jd_text, str) or not jd_text.strip():
        return {
            "required_skills": set(),
            "clean_text": ""
        }
        
    # Extract skills using the pre-defined skill dictionary patterns
    required_skills = extract_skills_from_text(jd_text)
    
    return {
        "required_skills": required_skills,
        "clean_text": jd_text
    }

if __name__ == "__main__":
    test_jd = "Looking for a Data Scientist with 3+ years experience. Required skills: Python, PyTorch, SQL, AWS, and Machine Learning."
    parsed = parse_job_description(test_jd)
    print("Job Description:", test_jd)
    print("Parsed Skills:", list(parsed["required_skills"]))

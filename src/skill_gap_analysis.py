def perform_skill_gap_analysis(candidate_skills, required_skills):
    """
    Computes matched and missing skills, and skill match percentage for a candidate.
    
    Parameters:
    candidate_skills (set): Set of skills extracted from candidate resume.
    required_skills (set): Set of skills required by the job description.
    
    Returns:
    dict: A dictionary containing matched_skills (list), missing_skills (list), and skill_match_pct (float).
    """
    # Ensure inputs are sets
    candidate_skills = set(candidate_skills) if candidate_skills else set()
    required_skills = set(required_skills) if required_skills else set()
    
    if not required_skills:
        # If job description contains no detectable required skills, skill match is 0.0 or 100.0?
        # Standard approach is to return 0.0 or handle gracefully. Let's return 0.0 and specify 
        # that no required skills were detected.
        return {
            "matched_skills": [],
            "missing_skills": [],
            "skill_match_pct": 0.0
        }
        
    matched = candidate_skills.intersection(required_skills)
    missing = required_skills.difference(candidate_skills)
    
    skill_match_pct = (len(matched) / len(required_skills)) * 100.0
    
    return {
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "skill_match_pct": round(skill_match_pct, 2)
    }

import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set writeable path for NLTK data in serverless environments (like Vercel)
import os
nltk_data_dir = "/tmp/nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Initialize NLTK resources automatically if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)

# Attempt to load spaCy, with fallback
nlp = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    # Fallback to NLTK WordNet
    nlp = None

# Custom list of technical stopwords that are generic but shouldn't interfere with skills
TECHNICAL_EXCLUSIONS = {
    "work", "experience", "project", "company", "team", "responsibilities", "system", 
    "management", "development", "business", "application", "technology", "client", "support"
}

def clean_text(text):
    """
    Cleans raw resume or job description text.
    Removes HTML tags, URLs, email addresses, phone numbers, punctuation, 
    and handles HTML entities while preserving technical terms like C++, C#, .NET.
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities (e.g. &amp; -> &)
    text = html.unescape(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Remove Email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove Phone numbers (common patterns)
    text = re.sub(r'\+?\d[\d\-\(\)\s]{8,}\d', ' ', text)
    
    # Remove unnecessary punctuation but preserve C++, C#, .NET, Node.js, React.js
    # Let's replace tabs/newlines with spaces first
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Keep only alphanumeric characters, spaces, and selected technical symbols (+, #, ., -)
    # We do a custom regex filter
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.\-]', ' ', text)
    
    # Reduce multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """
    NLP preprocessing pipeline: Cleans text, tokenizes, removes standard stopwords,
    and lemmatizes remaining words. Automatically falls back if spaCy is not installed.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""
        
    stop_words = set(stopwords.words('english'))
    # Do not remove 'c' since it can refer to C programming language, 
    # but remove other generic letters.
    if 'c' in stop_words:
        stop_words.remove('c')
        
    # Standardize spaces around technical terms like C++ to avoid merging
    # e.g., "python,c++" -> "python c++"
    cleaned = re.sub(r'(?<=[a-zA-Z])\+', ' +', cleaned)
    cleaned = re.sub(r'\+(?=[a-zA-Z])', '+ ', cleaned)
    
    if nlp is not None:
        # Use spaCy for lemmatization
        doc = nlp(cleaned)
        tokens = []
        for token in doc:
            lemma = token.lemma_.strip()
            # Preserve special technical tags
            if lemma in ["c++", "c#", ".net"]:
                tokens.append(lemma)
            elif lemma not in stop_words and len(lemma) > 1 and lemma.isalpha():
                tokens.append(lemma)
        return " ".join(tokens)
    else:
        # Fallback to NLTK
        lemmatizer = WordNetLemmatizer()
        raw_tokens = cleaned.split()
        tokens = []
        for t in raw_tokens:
            # Preserve special technical tokens as-is
            if t in ["c++", "c#", ".net"]:
                tokens.append(t)
            elif t not in stop_words and len(t) > 1:
                # Lemmatize verbs, nouns, adjectives
                lemma = lemmatizer.lemmatize(t, pos='v')
                lemma = lemmatizer.lemmatize(lemma, pos='n')
                if lemma.isalpha():
                    tokens.append(lemma)
        return " ".join(tokens)

if __name__ == "__main__":
    test_str = "We are seeking a Machine Learning Engineer proficient in Python, C++, C#, .NET, Node.js, and SQL."
    print("Original:", test_str)
    print("Preprocessed:", preprocess_text(test_str))

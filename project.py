from pypdf import PdfReader
import docx

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

# matcher.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Common tech skills database (can be expanded)
COMMON_SKILLS = [
    "python", "java", "c++", "javascript", "typescript", "react", "node.js",
    "sql", "nosql", "mongodb", "postgresql", "docker", "kubernetes", "aws",
    "git", "machine learning", "deep learning", "nlp", "html", "css",
    "fastapi", "django", "flask", "rest api", "data analysis", "pandas"
]

def extract_skills(text: str) -> set:
    text_lower = text.lower()
    found_skills = set()
    for skill in COMMON_SKILLS:
        # Match as whole word/phrase
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return found_skills

def calculate_match_score(resume_text: str, job_description: str) -> float:
    # Uses TF-IDF cosine similarity between resume and job description
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)
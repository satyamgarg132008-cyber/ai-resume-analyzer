import os
import re
import streamlit as st
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Resume Analyzer & Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & ANIMATIONS ---
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 10px rgba(168, 85, 247, 0.2); }
        50% { box-shadow: 0 0 25px rgba(6, 182, 212, 0.4); }
        100% { box-shadow: 0 0 10px rgba(168, 85, 247, 0.2); }
    }
    .main-card {
        animation: fadeIn 0.8s ease-out;
        padding: 1.5rem;
        border-radius: 1rem;
        background: rgba(22, 31, 48, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    .metric-container {
        animation: pulseGlow 3s infinite ease-in-out;
        border-radius: 1rem;
        padding: 1.2rem;
        text-align: center;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-matched {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 9999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-missing {
        display: inline-block;
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 9999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SKILLS DATABASE ---
COMMON_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "react", "angular",
    "vue", "node.js", "express", "sql", "nosql", "mongodb", "postgresql", "mysql",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "github", "machine learning",
    "deep learning", "nlp", "html", "css", "fastapi", "django", "flask",
    "rest api", "graphql", "data analysis", "pandas", "numpy", "power bi", "tableau",
    "linux", "ci/cd", "agile", "scrum", "microservices", "unit testing"
]

# --- EXTRACTION FUNCTIONS ---
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

def extract_skills(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found

def calculate_match_score(resume_text: str, job_desc: str) -> float:
    if not resume_text.strip() or not job_desc.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(sim) * 100, 1)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/resume.png", width=70)
    st.title("About Project")
    st.info(
        "**AI Resume Analyzer & Matcher**\n\n"
        "Evaluates how well a candidate's resume matches a target job description using NLP, TF-IDF cosine similarity, and skill extraction."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Quick Guide")
    st.markdown("1. Upload candidate resume (PDF or DOCX)")
    st.markdown("2. Paste target Job Description")
    st.markdown("3. Click **Analyze Resume**")
    st.markdown("---")
    st.caption("Built for Streamlit Community Cloud")

# --- MAIN CONTENT ---
st.title("📄 AI-Based Resume Analyzer")
st.markdown("Evaluate candidate-job fit with intelligent keyword matching, skill gap discovery, and similarity scoring.")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    resume_text = ""
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            st.success(f"✅ Loaded: **{uploaded_file.name}** ({len(resume_text.split())} words)")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

with col_right:
    st.subheader("2. Target Job Description")
    job_desc = st.text_area(
        "Paste the Job Description / Requirements",
        height=200,
        placeholder="e.g. We are looking for a Python Developer experienced in React, PostgreSQL, Docker, and REST APIs..."
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🚀 Analyze Fit & Generate Report", type="primary", use_container_width=True)

if analyze_btn:
    if not resume_text:
        st.warning("⚠️ Please upload a resume first.")
    elif not job_desc.strip():
        st.warning("⚠️ Please provide a job description.")
    else:
        with st.spinner("🤖 Extracting skills and calculating match similarity..."):
            score = calculate_match_score(resume_text, job_desc)
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_desc)
            matched_skills = sorted(list(resume_skills.intersection(job_skills)))
            missing_skills = sorted(list(job_skills.difference(resume_skills)))

        st.markdown("---")
        st.subheader("📊 Candidate Evaluation Dashboard")

        # Top metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Similarity Score", f"{score}%")
        m2.metric("Matched Skills", len(matched_skills))
        m3.metric("Missing Skills", len(missing_skills))
        m4.metric("Total Skills Found", len(resume_skills))

        st.progress(min(int(score), 100))

        if score >= 70:
            st.balloons()
            st.success("🌟 **High Match:** This candidate possesses the key competencies required for this role!")
        elif score >= 40:
            st.info("👍 **Moderate Match:** The candidate has relevant skills but is missing a few core qualifications.")
        else:
            st.warning("⚠️ **Low Match:** Major skill gaps identified between the resume and job requirements.")

        # Skills breakdown
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("### ✅ Matched Skills")
            if matched_skills:
                badges_html = "".join([f'<span class="badge-matched">{s.title()}</span>' for s in matched_skills])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.write("No direct skill matches found.")

        with sc2:
            st.markdown("### ⚠️ Missing Required Skills")
            if missing_skills:
                badges_html = "".join([f'<span class="badge-missing">{s.title()}</span>' for s in missing_skills])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.write("No missing skills detected! Perfect skill coverage.")

        # Download Report
        report_text = f"""=====================================
AI RESUME EVALUATION REPORT
=====================================
Match Score: {score}%
Total Resume Words: {len(resume_text.split())}
Skills Matched ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None'}
Missing Skills ({len(missing_skills)}): {', '.join(missing_skills) if missing_skills else 'None'}
=====================================
"""
        st.download_button(
            label="📥 Download Audit Report (.txt)",
            data=report_text,
            file_name="resume_evaluation_report.txt",
            mime="text/plain"
        )

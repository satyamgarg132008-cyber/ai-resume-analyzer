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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN HIGH-TECH NEON / GLASSMORPHISM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Ambient Background Mesh */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 10% 15%, rgba(168, 85, 247, 0.18) 0%, transparent 45%),
                    radial-gradient(circle at 90% 85%, rgba(6, 182, 212, 0.18) 0%, transparent 45%),
                    radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.12) 0%, transparent 55%),
                    #0b0f19 !important;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #080c14 100%) !important;
        border-right: 1px solid rgba(168, 85, 247, 0.25) !important;
    }

    /* Glowing Gradient Title */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #c084fc 0%, #38bdf8 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }

    /* Neon Gradient Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #9333ea 0%, #ec4899 50%, #06b6d4 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 0.9rem 2.5rem !important;
        box-shadow: 0 4px 25px rgba(168, 85, 247, 0.5) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 35px rgba(236, 72, 153, 0.7) !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 1.25rem !important;
        padding: 1.2rem !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(12px) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Form Fields & Text Area */
    [data-testid="stTextArea"] textarea {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 1rem !important;
        color: #f8fafc !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3) !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 2px dashed rgba(168, 85, 247, 0.4) !important;
        border-radius: 1.2rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.25) !important;
    }

    /* Vibrant Skill Pills */
    .pill-match {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.15) 100%);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.5);
        border-radius: 9999px;
        padding: 0.4rem 1rem;
        font-size: 0.88rem;
        font-weight: 700;
        margin: 0.3rem;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }
    .pill-miss {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.25) 0%, rgba(225, 29, 72, 0.15) 100%);
        color: #fb7185;
        border: 1px solid rgba(251, 113, 133, 0.5);
        border-radius: 9999px;
        padding: 0.4rem 1rem;
        font-size: 0.88rem;
        font-weight: 700;
        margin: 0.3rem;
        box-shadow: 0 0 12px rgba(244, 63, 94, 0.2);
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

# --- SIDEBAR BRANDING ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 70px; height: 70px; border-radius: 1.5rem; background: linear-gradient(135deg, #a855f7, #06b6d4); box-shadow: 0 0 30px rgba(168, 85, 247, 0.5);">
            <span style="font-size: 2.2rem;">⚡</span>
        </div>
        <h2 style="margin: 0.8rem 0 0.2rem 0; font-weight: 900; background: linear-gradient(135deg, #c084fc, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ResumeAI</h2>
        <span style="display: inline-block; padding: 0.2rem 0.8rem; border-radius: 9999px; background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.4); font-size: 0.75rem; font-weight: 700; color: #c084fc; letter-spacing: 1px; text-transform: uppercase;">Pro Edition</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 1rem; padding: 1rem; margin-bottom: 1.5rem;">
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin: 0;">
            🚀 <b>Intelligent Screening Engine</b> powered by TF-IDF vectorization and semantic keyword entity matching.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Workflow")
    st.markdown("1️⃣ **Upload Resume** (PDF/DOCX)")
    st.markdown("2️⃣ **Add Target Job Specs**")
    st.markdown("3️⃣ **Run AI Diagnostics**")
    st.markdown("---")
    st.caption("✨ Deployed live via Streamlit Community Cloud")

# --- MAIN HERO HEADER ---
st.markdown('<h1 class="hero-title">⚡ AI Resume Matcher & Diagnostic</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Evaluate candidate alignment, quantify skill gaps, and optimize resumes with instant AI scoring.</p>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### 📁 1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload PDF or Word document", type=["pdf", "docx"])
    resume_text = ""
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            st.success(f"✨ **{uploaded_file.name}** parsed successfully ({len(resume_text.split())} words)")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

with col_right:
    st.markdown("### 📋 2. Target Job Description")
    job_desc = st.text_area(
        "Paste Job Requirements / Description",
        height=210,
        placeholder="Paste role requirements here... (e.g. Seeking Python Developer with experience in React, Docker, FastAPI, AWS, and PostgreSQL...)"
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("⚡ Run AI Analysis & Fit Check", use_container_width=True)

if analyze_btn:
    if not resume_text:
        st.warning("⚠️ Please upload a resume first.")
    elif not job_desc.strip():
        st.warning("⚠️ Please provide a job description.")
    else:
        with st.spinner("🔮 Scanning resume entities, calculating vectors, and matching skills..."):
            score = calculate_match_score(resume_text, job_desc)
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_desc)
            matched_skills = sorted(list(resume_skills.intersection(job_skills)))
            missing_skills = sorted(list(job_skills.difference(resume_skills)))

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;'>", unsafe_allow_html=True)

        # Big Hero Score Card
        if score >= 70:
            badge_color = "#34d399"
            verdict = "🌟 Outstanding Match — Highly Recommended for Interview"
            st.balloons()
        elif score >= 40:
            badge_color = "#38bdf8"
            verdict = "👍 Promising Fit — Moderate Skill Coverage with minor gaps"
        else:
            badge_color = "#fb7185"
            verdict = "⚠️ Low Compatibility — Critical Skill Gaps Identified"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(6, 182, 212, 0.15)); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 1.5rem; padding: 2.2rem; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(168, 85, 247, 0.25);">
            <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; color: #94a3b8; font-weight: 700; margin-bottom: 0.5rem;">Overall Compatibility Score</div>
            <div style="font-size: 4.8rem; font-weight: 900; background: linear-gradient(135deg, #c084fc, #38bdf8, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;">{score}%</div>
            <div style="font-size: 1.15rem; color: {badge_color}; margin-top: 0.8rem; font-weight: 700;">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Match Score", f"{score}%")
        m2.metric("Matched Skills", len(matched_skills))
        m3.metric("Missing Skills", len(missing_skills))
        m4.metric("Total Skills in Resume", len(resume_skills))

        st.progress(min(int(score), 100))

        st.markdown("<br>", unsafe_allow_html=True)

        # Skill Badges
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 1.25rem; padding: 1.5rem; min-height: 200px;">
                <h3 style="color: #34d399; font-weight: 800; margin-top: 0; display: flex; align-items: center; gap: 8px;">
                    ✅ Matched Skills ({len(matched_skills)})
                </h3>
            """, unsafe_allow_html=True)
            if matched_skills:
                badges_html = "".join([f'<span class="pill-match">✓ {s.title()}</span>' for s in matched_skills])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #64748b;'>No direct skill matches detected.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with sc2:
            st.markdown("""
            <div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 1.25rem; padding: 1.5rem; min-height: 200px;">
                <h3 style="color: #fb7185; font-weight: 800; margin-top: 0; display: flex; align-items: center; gap: 8px;">
                    ⚠️ Missing / Required Skills ({len(missing_skills)})
                </h3>
            """, unsafe_allow_html=True)
            if missing_skills:
                badges_html = "".join([f'<span class="pill-miss">✕ {s.title()}</span>' for s in missing_skills])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #34d399;'>All required skills found in resume!</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Download Report
        report_text = f"""=====================================
AI RESUME COMPATIBILITY AUDIT REPORT
=====================================
Match Score: {score}%
Verdict: {verdict}
Total Resume Word Count: {len(resume_text.split())}
Total Candidate Skills Detected: {len(resume_skills)}

MATCHED SKILLS ({len(matched_skills)}):
{', '.join(matched_skills) if matched_skills else 'None'}

MISSING SKILLS ({len(missing_skills)}):
{', '.join(missing_skills) if missing_skills else 'None'}
=====================================
Generated by AI Resume Matcher
"""
        st.download_button(
            label="📥 Download Full Evaluation Report (.txt)",
            data=report_text,
            file_name="resume_evaluation_report.txt",
            mime="text/plain",
            use_container_width=True
        )

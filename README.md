# 📄 AI-Based Resume Analyzer

An intelligent candidate screening and skill-matching web application built with Python and Streamlit.

## 🚀 Features
- **PDF & DOCX Support:** Automatically parses text from `.pdf` and `.docx` resumes.
- **Skill Extraction:** Scans and extracts key technical competencies.
- **Match Scoring:** Computes similarity score using TF-IDF and Cosine Similarity against target Job Descriptions.
- **Gap Analysis:** Identifies matched skills and missing qualifications.
- **Export Report:** Downloadable candidate evaluation report.

## 💻 Local Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open in your browser: `http://localhost:8501`

## ☁️ How to Deploy on Streamlit Community Cloud
1. Push this folder to a public **GitHub** repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **New app**, select your repo, set Main file path to `app.py`.
4. Click **Deploy!**

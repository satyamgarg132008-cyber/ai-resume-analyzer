import os
import re
from flask import Flask, request, jsonify, render_template_string
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

COMMON_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "react", "angular",
    "vue", "node.js", "express", "sql", "nosql", "mongodb", "postgresql", "mysql",
    "docker", "kubernetes", "aws", "azure", "git", "github", "machine learning",
    "deep learning", "nlp", "html", "css", "fastapi", "django", "flask",
    "rest api", "data analysis", "pandas", "numpy", "power bi", "tableau", "linux"
]

def extract_text_from_pdf(file_stream) -> str:
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file_stream) -> str:
    doc = docx.Document(file_stream)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def extract_skills(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found

def calculate_match(resume_text: str, job_desc: str) -> float:
    if not resume_text.strip() or not job_desc.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume_text, job_desc])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(float(sim) * 100, 1)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400
    
    file = request.files['resume']
    job_desc = request.form.get('job_description', '')

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(file.stream)
    elif filename.endswith('.docx'):
        resume_text = extract_text_from_docx(file.stream)
    else:
        return jsonify({"error": "Please upload a .pdf or .docx file"}), 400

    score = calculate_match(resume_text, job_desc)
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)
    
    matched = sorted(list(resume_skills.intersection(job_skills)))
    missing = sorted(list(job_skills.difference(resume_skills)))

    return jsonify({
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "word_count": len(resume_text.split()),
        "total_resume_skills": len(resume_skills)
    })

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Resume Analyzer (Animated)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @keyframes floatOrb {
            0%, 100% { transform: translateY(0px) scale(1); }
            50% { transform: translateY(-30px) scale(1.1); }
        }
        @keyframes laserScan {
            0% { top: 0%; opacity: 0; }
            30% { opacity: 1; }
            70% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }
        @keyframes badgePop {
            0% { transform: scale(0.6); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }
        .float-1 { animation: floatOrb 8s ease-in-out infinite; }
        .float-2 { animation: floatOrb 10s ease-in-out infinite reverse; }
        .scan-line { animation: laserScan 2s linear infinite; }
        .badge-anim { animation: badgePop 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards; }
        
        .circle-progress {
            stroke-dasharray: 440;
            stroke-dashoffset: 440;
            transition: stroke-dashoffset 1.8s ease-out;
        }
    </style>
</head>
<body class="bg-[#0b0f19] text-gray-100 min-h-screen relative overflow-x-hidden font-sans">
    <div class="fixed top-[-100px] left-[-100px] w-96 h-96 bg-purple-600/30 rounded-full blur-[120px] pointer-events-none float-1"></div>
    <div class="fixed bottom-[-100px] right-[-100px] w-96 h-96 bg-cyan-500/25 rounded-full blur-[120px] pointer-events-none float-2"></div>

    <div class="max-w-6xl mx-auto px-6 py-10 relative z-10">
        <header class="text-center mb-10">
            <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-sm mb-4">
                <i class="fa-solid fa-sparkles text-xs"></i> Intelligent Candidate Scoring
            </div>
            <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-purple-400 via-cyan-300 to-indigo-300 bg-clip-text text-transparent">
                AI Resume Matcher & Analyzer
            </h1>
            <p class="text-gray-400 mt-2 text-base">Instant document scanning, keyword matching, and skill gap detection.</p>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
            <div class="bg-gray-900/60 backdrop-blur-xl border border-gray-800 rounded-2xl p-6 shadow-2xl relative group hover:border-purple-500/40 transition-all duration-300">
                <h2 class="text-lg font-semibold flex items-center gap-2 mb-4 text-purple-300">
                    <i class="fa-solid fa-file-arrow-up"></i> 1. Upload Resume
                </h2>
                <label id="dropzone" class="flex flex-col items-center justify-center border-2 border-dashed border-gray-700 hover:border-purple-400 rounded-xl p-8 cursor-pointer transition-all duration-300 bg-gray-950/40 hover:bg-purple-950/10 group-hover:shadow-inner">
                    <i class="fa-solid fa-cloud-arrow-up text-4xl text-purple-400 mb-3 group-hover:scale-110 transition-transform"></i>
                    <span id="fileNameDisplay" class="font-medium text-gray-300">Choose PDF or DOCX</span>
                    <span class="text-xs text-gray-500 mt-1">Drag and drop or click to browse</span>
                    <input type="file" id="resumeInput" accept=".pdf,.docx" class="hidden">
                </label>
            </div>

            <div class="bg-gray-900/60 backdrop-blur-xl border border-gray-800 rounded-2xl p-6 shadow-2xl hover:border-cyan-500/40 transition-all duration-300">
                <h2 class="text-lg font-semibold flex items-center gap-2 mb-4 text-cyan-300">
                    <i class="fa-solid fa-briefcase"></i> 2. Target Job Description
                </h2>
                <textarea id="jobDescInput" rows="5" class="w-full bg-gray-950/60 border border-gray-800 rounded-xl p-4 text-sm text-gray-200 focus:outline-none focus:border-cyan-400 transition" placeholder="Paste the target job description or requirements here..."></textarea>
            </div>
        </div>

        <div class="flex justify-center mb-12">
            <button id="analyzeBtn" onclick="runAnalysis()" class="relative group px-10 py-4 rounded-xl font-bold text-white bg-gradient-to-r from-purple-600 via-indigo-600 to-cyan-500 hover:opacity-95 shadow-lg shadow-purple-600/30 hover:shadow-cyan-500/40 active:scale-95 transition-all duration-200 flex items-center gap-3">
                <i class="fa-solid fa-wand-magic-sparkles text-lg"></i>
                <span id="btnText">Analyze Fit & Match</span>
            </button>
        </div>

        <div id="scannerModal" class="hidden fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex flex-col items-center justify-center">
            <div class="relative w-64 h-80 bg-gray-900 border border-purple-500/50 rounded-2xl shadow-2xl overflow-hidden p-6 flex flex-col justify-between">
                <div class="absolute inset-x-0 h-1 bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 shadow-[0_0_15px_#a855f7] scan-line"></div>
                <div class="space-y-3">
                    <div class="h-4 bg-gray-800 rounded w-3/4 animate-pulse"></div>
                    <div class="h-3 bg-gray-800 rounded w-full animate-pulse"></div>
                    <div class="h-3 bg-gray-800 rounded w-5/6 animate-pulse"></div>
                    <div class="h-3 bg-gray-800 rounded w-2/3 animate-pulse"></div>
                </div>
                <div class="text-center">
                    <p class="text-sm font-semibold text-purple-300 animate-pulse">Scanning Document...</p>
                    <p class="text-xs text-gray-500 mt-1">Extracting skills & entities</p>
                </div>
            </div>
        </div>

        <div id="resultsSection" class="hidden transition-all duration-500">
            <div class="bg-gray-900/80 backdrop-blur-2xl border border-gray-800 rounded-3xl p-8 shadow-2xl">
                <h2 class="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                    Evaluation Dashboard
                </h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 items-center mb-10">
                    <div class="flex flex-col items-center justify-center bg-gray-950/60 border border-gray-800 rounded-2xl p-6">
                        <div class="relative w-40 h-40 flex items-center justify-center">
                            <svg class="w-full h-full -rotate-90" viewBox="0 0 160 160">
                                <circle cx="80" cy="80" r="70" stroke="currentColor" stroke-width="12" class="text-gray-800" fill="transparent"/>
                                <circle id="circleGauge" cx="80" cy="80" r="70" stroke="url(#gradient)" stroke-width="12" fill="transparent" stroke-linecap="round" class="circle-progress"/>
                                <defs>
                                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stop-color="#a855f7"/>
                                        <stop offset="100%" stop-color="#06b6d4"/>
                                    </linearGradient>
                                </defs>
                            </svg>
                            <div class="absolute text-center">
                                <span id="scoreText" class="text-4xl font-black text-white">0%</span>
                                <span class="block text-xs uppercase tracking-wider text-gray-400">Match Score</span>
                            </div>
                        </div>
                    </div>
                    <div class="bg-gray-950/60 border border-gray-800 rounded-2xl p-6 text-center">
                        <i class="fa-solid fa-circle-check text-emerald-400 text-3xl mb-2"></i>
                        <span id="matchedCount" class="block text-4xl font-extrabold text-emerald-400">0</span>
                        <span class="text-sm text-gray-400">Skills Matched</span>
                    </div>
                    <div class="bg-gray-950/60 border border-gray-800 rounded-2xl p-6 text-center">
                        <i class="fa-solid fa-triangle-exclamation text-rose-400 text-3xl mb-2"></i>
                        <span id="missingCount" class="block text-4xl font-extrabold text-rose-400">0</span>
                        <span class="text-sm text-gray-400">Missing Skills</span>
                    </div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="bg-emerald-950/20 border border-emerald-500/30 rounded-2xl p-6">
                        <h3 class="text-emerald-400 font-semibold mb-4 flex items-center gap-2">
                            <i class="fa-solid fa-check"></i> Matched Candidate Skills
                        </h3>
                        <div id="matchedBadges" class="flex flex-wrap gap-2"></div>
                    </div>
                    <div class="bg-rose-950/20 border border-rose-500/30 rounded-2xl p-6">
                        <h3 class="text-rose-400 font-semibold mb-4 flex items-center gap-2">
                            <i class="fa-solid fa-xmark"></i> Skills to Add / Missing
                        </h3>
                        <div id="missingBadges" class="flex flex-wrap gap-2"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('resumeInput');
        const fileNameDisplay = document.getElementById('fileNameDisplay');

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileNameDisplay.textContent = e.target.files[0].name;
                fileNameDisplay.classList.add('text-purple-300');
            }
        });

        async function runAnalysis() {
            const file = fileInput.files[0];
            const jobDesc = document.getElementById('jobDescInput').value.trim();

            if (!file) {
                alert('Please upload a resume file (PDF or DOCX).');
                return;
            }
            if (!jobDesc) {
                alert('Please enter or paste the target job description.');
                return;
            }

            const scanner = document.getElementById('scannerModal');
            scanner.classList.remove('hidden');

            const formData = new FormData();
            formData.append('resume', file);
            formData.append('job_description', jobDesc);

            try {
                const [response] = await Promise.all([
                    fetch('/api/analyze', { method: 'POST', body: formData }),
                    new Promise(resolve => setTimeout(resolve, 1500))
                ]);

                const data = await response.json();
                scanner.classList.add('hidden');

                if (data.error) {
                    alert(data.error);
                    return;
                }

                const resultsSection = document.getElementById('resultsSection');
                resultsSection.classList.remove('hidden');
                resultsSection.scrollIntoView({ behavior: 'smooth' });

                animateScore(data.score);
                document.getElementById('matchedCount').textContent = data.matched_skills.length;
                document.getElementById('missingCount').textContent = data.missing_skills.length;

                renderBadges('matchedBadges', data.matched_skills, 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40');
                renderBadges('missingBadges', data.missing_skills, 'bg-rose-500/20 text-rose-300 border-rose-500/40');

                if (data.score >= 50) {
                    confetti({ particleCount: 120, spread: 80, origin: { y: 0.6 } });
                }

            } catch (err) {
                scanner.classList.add('hidden');
                alert('Analysis failed: ' + err.message);
            }
        }

        function animateScore(targetScore) {
            const circle = document.getElementById('circleGauge');
            const scoreText = document.getElementById('scoreText');
            const circumference = 2 * Math.PI * 70;
            const offset = circumference - (targetScore / 100) * circumference;
            circle.style.strokeDashoffset = offset;

            let current = 0;
            const interval = setInterval(() => {
                if (current >= targetScore) {
                    scoreText.textContent = targetScore + '%';
                    clearInterval(interval);
                } else {
                    current += 1;
                    scoreText.textContent = current + '%';
                }
            }, 18);
        }

        function renderBadges(containerId, skills, styleClasses) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            if (skills.length === 0) {
                container.innerHTML = '<span class="text-gray-500 text-sm">None detected</span>';
                return;
            }
            skills.forEach((skill, i) => {
                const badge = document.createElement('span');
                badge.className = `px-3 py-1.5 rounded-lg border text-xs font-semibold badge-anim capitalize ${styleClasses}`;
                badge.style.animationDelay = `${i * 60}ms`;
                badge.textContent = skill;
                container.appendChild(badge);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🚀 Starting AI Resume Analyzer at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

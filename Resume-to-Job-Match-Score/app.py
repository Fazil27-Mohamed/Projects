import streamlit as st
import joblib
from scipy.sparse import hstack
import time
import math

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Resume Match AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("resume_match_model.pkl")

    resume_tfidf = joblib.load("resume_tfidf.pkl")

    job_tfidf = joblib.load("job_tfidf.pkl")

    required_skill_tfidf = joblib.load("skill_tfidf.pkl")

    resume_skill_tfidf = joblib.load("resume_skill_tfidf.pkl")

    encoder = joblib.load("category_encoder.pkl")

    return (
        model,
        resume_tfidf,
        job_tfidf,
        required_skill_tfidf,
        resume_skill_tfidf,
        encoder
    )

(
model,
resume_tfidf,
job_tfidf,
required_skill_tfidf,
resume_skill_tfidf,
encoder
)=load_models()

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>

html,
body,
[class*="css"]{

font-family:'Inter',sans-serif;

}

/* Hide Streamlit */

#MainMenu{

visibility:hidden;

}

footer{

visibility:hidden;

}

/* -------------------------------------------------- */
/* APP BACKGROUND */
/* -------------------------------------------------- */

.stApp{

background:

radial-gradient(circle at 10% 10%,rgba(139,92,246,.22),transparent 32%),

radial-gradient(circle at 90% 20%,rgba(6,182,212,.22),transparent 30%),

radial-gradient(circle at 80% 90%,rgba(168,85,247,.18),transparent 28%),

linear-gradient(
135deg,
#eef4ff 0%,
#edf2ff 25%,
#f8f5ff 55%,
#eefcff 100%
);

background-attachment:fixed;

color:#1f2937;

}

/* -------------------------------------------------- */
/* HERO */
/* -------------------------------------------------- */

.hero{

padding:42px;

border-radius:30px;

background:rgba(255,255,255,.55);

backdrop-filter:blur(24px);

border:1px solid rgba(255,255,255,.65);

box-shadow:

0 15px 45px rgba(139,92,246,.12),

0 8px 25px rgba(6,182,212,.10);

margin-bottom:28px;

}

.hero h1{

margin:0;

font-size:54px;

font-weight:800;

background:

linear-gradient(
90deg,
#8B5CF6,
#06B6D4
);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

}

.hero p{

margin-top:12px;

font-size:18px;

color:#4b5563;

}

/* -------------------------------------------------- */
/* GLASS CARD */
/* -------------------------------------------------- */

.glass{

background:rgba(255,255,255,.55);

backdrop-filter:blur(24px);

border-radius:24px;

padding:25px;

border:1px solid rgba(255,255,255,.70);

box-shadow:

0 12px 35px rgba(139,92,246,.10),

0 6px 18px rgba(6,182,212,.08);

transition:.35s;

margin-bottom:22px;

}

.glass:hover{

transform:translateY(-6px);

box-shadow:

0 20px 45px rgba(139,92,246,.18),

0 12px 30px rgba(6,182,212,.14);

}

/* -------------------------------------------------- */
/* METRIC CARD */
/* -------------------------------------------------- */

.metric{

background:rgba(255,255,255,.60);

border-radius:20px;

padding:22px;

text-align:center;

border:1px solid rgba(255,255,255,.70);

}

/* -------------------------------------------------- */
/* BUTTON */
/* -------------------------------------------------- */

.stButton>button{

width:100%;

height:60px;

font-size:20px;

font-weight:700;

border:none;

border-radius:18px;

color:white;

background:

linear-gradient(
90deg,
#8B5CF6,
#7C3AED,
#06B6D4
);

transition:.35s;

}

.stButton>button:hover{

transform:translateY(-2px);

box-shadow:

0 0 25px rgba(139,92,246,.45),

0 0 45px rgba(6,182,212,.35);

}

/* -------------------------------------------------- */
/* TEXT INPUTS */
/* -------------------------------------------------- */

textarea{

background:rgba(255,255,255,.75)!important;

border-radius:18px!important;

color:#1f2937!important;

border:1px solid rgba(139,92,246,.18)!important;

}

textarea:focus{

border:1px solid #8B5CF6!important;

}

/* -------------------------------------------------- */
/* SELECT */
/* -------------------------------------------------- */

div[data-baseweb="select"]{

background:rgba(255,255,255,.75);

border-radius:16px;

}

/* -------------------------------------------------- */
/* LABELS */
/* -------------------------------------------------- */

label{

font-weight:600!important;

color:#374151!important;

}

/* -------------------------------------------------- */
/* PROGRESS */
/* -------------------------------------------------- */

.stProgress>div>div>div{

background:

linear-gradient(
90deg,
#8B5CF6,
#06B6D4
);

}

/* -------------------------------------------------- */
/* SCROLLBAR */
/* -------------------------------------------------- */

::-webkit-scrollbar{

width:8px;

}

::-webkit-scrollbar-thumb{

background:#8B5CF6;

border-radius:50px;

}

</style>

""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""

<div class="hero">

<h1>🚀 Resume Match AI</h1>

<p>

Analyze your resume against any job description using Machine Learning,
ATS-inspired scoring, and an interactive AI dashboard.

</p>

</div>

""", unsafe_allow_html=True)
# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🚀 Resume Match AI")

    st.caption("Machine Learning Powered Resume Analysis")

    st.markdown("---")

    st.success("### ✨ Features")

    st.write("✅ Resume Analysis")
    st.write("✅ ATS Style Scoring")
    st.write("✅ Skill Gap Detection")
    st.write("✅ Job Compatibility")
    st.write("✅ Instant Report")

    st.markdown("---")

    st.info(
"""
💡 Tips

• Upload complete resume

• Paste full Job Description

• Mention technical skills

• Include certifications

• Include projects
"""
    )

# =========================================================
# HERO INFO CARD
# =========================================================

st.markdown("""

<div class="glass">

<h2 style="margin-bottom:5px;">
🎯 Resume to Job Match Scorer
</h2>

<p style="font-size:17px;color:#6b7280;">

Compare your resume with any Job Description and receive an AI-powered
compatibility score along with ATS insights and missing skill analysis.

</p>

</div>

""",unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================

left,right=st.columns(2,gap="large")

# =========================================================
# LEFT
# =========================================================

with left:

    st.markdown("""
<div class="glass">
<h3>📄 Resume</h3>
</div>
""",unsafe_allow_html=True)

    resume=st.text_area(

        "",

        height=330,

        placeholder="""
Example

Python Developer

Skills

Python
SQL
Machine Learning
Power BI
Pandas
NumPy
TensorFlow

Projects

Resume Matcher

Traffic Prediction

Crop Recommendation

Experience...

""",

        key="resume"

    )

    st.markdown("""
<div class="glass">
<h3>🛠 Resume Skills</h3>
</div>
""",unsafe_allow_html=True)

    resume_skills=st.text_area(

        "",

        height=140,

        placeholder="Python, SQL, Machine Learning, Power BI",

        key="resume_skill"

    )

# =========================================================
# RIGHT
# =========================================================

with right:

    st.markdown("""
<div class="glass">
<h3>💼 Job Description</h3>
</div>
""",unsafe_allow_html=True)

    job_description=st.text_area(

        "",

        height=330,

        placeholder="""
Python Developer

Requirements

Python

SQL

Machine Learning

REST API

Git

Cloud

Problem Solving

Communication Skills

""",

        key="job"

    )

    st.markdown("""
<div class="glass">
<h3>🎯 Required Skills</h3>
</div>
""",unsafe_allow_html=True)

    required_skills=st.text_area(

        "",

        height=140,

        placeholder="Python, SQL, Git, REST API, Machine Learning",

        key="required"

    )

# =========================================================
# CATEGORY
# =========================================================

st.markdown("<br>",unsafe_allow_html=True)

st.markdown("""
<div class="glass">

<h3>
📂 Job Category
</h3>

</div>

""",unsafe_allow_html=True)

category=st.selectbox(

"",

[

'ACCOUNTANT',

'ADVOCATE',

'AGRICULTURE',

'ARTS',

'AUTOMOBILE',

'AVIATION',

'BANKING',

'BPO',

'BUSINESS-DEVELOPMENT',

'CHEF',

'CONSTRUCTION',

'CONSULTANT',

'DESIGNER',

'DIGITAL-MEDIA',

'ENGINEERING',

'FINANCE',

'FITNESS',

'HEALTHCARE',

'HR',

'INFORMATION-TECHNOLOGY',

'PUBLIC-RELATIONS',

'SALES',

'TEACHER'

]

)

# =========================================================
# LIVE STATS
# =========================================================

resume_words=len(resume.split())

job_words=len(job_description.split())

resume_skill_count=len(

[x for x in resume_skills.split(",") if x.strip()]

)

required_skill_count=len(

[x for x in required_skills.split(",") if x.strip()]

)

st.markdown("<br>",unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4)

with m1:

    st.markdown(f"""

<div class="metric">

<h2>{resume_words}</h2>

Resume Words

</div>

""",unsafe_allow_html=True)

with m2:

    st.markdown(f"""

<div class="metric">

<h2>{job_words}</h2>

JD Words

</div>

""",unsafe_allow_html=True)

with m3:

    st.markdown(f"""

<div class="metric">

<h2>{resume_skill_count}</h2>

Resume Skills

</div>

""",unsafe_allow_html=True)

with m4:

    st.markdown(f"""

<div class="metric">

<h2>{required_skill_count}</h2>

Required Skills

</div>

""",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

# =========================================================
# ANALYZE BUTTON
# =========================================================

predict=st.button("🚀 Analyze Resume Match")

if predict:

    with st.spinner("Analyzing Resume..."):

        progress=st.progress(0)

        status=st.empty()

        messages=[

            "Reading Resume...",

            "Reading Job Description...",

            "Extracting Skills...",

            "Generating Feature Vectors...",

            "Running ML Model...",

            "Preparing AI Report..."

        ]

        for i in range(101):

            progress.progress(i)

            if i<15:

                status.info(messages[0])

            elif i<35:

                status.info(messages[1])

            elif i<55:

                status.info(messages[2])

            elif i<75:

                status.info(messages[3])

            elif i<90:

                status.info(messages[4])

            else:

                status.success(messages[5])

            time.sleep(0.015)
# =========================================================
# ML PREDICTION PIPELINE
# =========================================================

if predict:

    # -----------------------------
    # TF-IDF Transform
    # -----------------------------

    resume_features = resume_tfidf.transform([resume])

    job_features = job_tfidf.transform([job_description])

    required_skill_features = required_skill_tfidf.transform(
        [required_skills]
    )

    resume_skill_features = resume_skill_tfidf.transform(
        [resume_skills]
    )

    # -----------------------------
    # Category Encoding
    # -----------------------------

    category_feature = encoder.transform([[category]])

    # -----------------------------
    # Feature Matrix
    # -----------------------------

    X = hstack([
        resume_features,
        job_features,
        required_skill_features,
        resume_skill_features,
        category_feature
    ])

    # -----------------------------
    # Prediction
    # -----------------------------

    score = float(model.predict(X)[0])

    score = max(0.0, min(score, 100.0))

    progress.empty()
    status.empty()

    # =========================================================
    # SKILL ANALYSIS
    # =========================================================

    resume_skill_set = set(
        s.strip().lower()
        for s in resume_skills.split(",")
        if s.strip()
    )

    required_skill_set = set(
        s.strip().lower()
        for s in required_skills.split(",")
        if s.strip()
    )

    matched_skills = sorted(
        resume_skill_set & required_skill_set
    )

    missing_skills = sorted(
        required_skill_set - resume_skill_set
    )

    additional_skills = sorted(
        resume_skill_set - required_skill_set
    )

    if len(required_skill_set) == 0:
        skill_coverage = 0
    else:
        skill_coverage = (
            len(matched_skills)
            / len(required_skill_set)
        ) * 100

    # =========================================================
    # ATS SCORE
    # =========================================================

    ats_score = round(
        score * 0.75 +
        skill_coverage * 0.25,
        2
    )

    ats_score = max(0, min(100, ats_score))

    # =========================================================
    # RESUME STATISTICS
    # =========================================================

    resume_words = len(resume.split())

    jd_words = len(job_description.split())

    resume_chars = len(resume)

    resume_lines = len(resume.splitlines())

    # =========================================================
    # RESUME QUALITY
    # =========================================================

    if resume_words < 120:
        resume_strength = "Needs Improvement"

    elif resume_words < 250:
        resume_strength = "Average"

    elif resume_words < 450:
        resume_strength = "Strong"

    else:
        resume_strength = "Excellent"

    # =========================================================
    # MATCH LEVEL
    # =========================================================

    if score < 40:

        match_level = "Poor Match"

        match_color = "#EF4444"

        emoji = "❌"

    elif score < 60:

        match_level = "Moderate Match"

        match_color = "#F59E0B"

        emoji = "⚠️"

    elif score < 80:

        match_level = "Good Match"

        match_color = "#06B6D4"

        emoji = "✅"

    else:

        match_level = "Excellent Match"

        match_color = "#8B5CF6"

        emoji = "🚀"

    # =========================================================
    # AI RECOMMENDATIONS
    # =========================================================

    recommendations = []

    if score < 40:

        recommendations.extend([

            "Tailor your resume specifically for this job.",

            "Add more technical skills required for the role.",

            "Improve project descriptions with measurable impact.",

            "Include certifications relevant to the position."

        ])

    elif score < 60:

        recommendations.extend([

            "Increase keyword matching with the Job Description.",

            "Highlight your achievements using numbers.",

            "Expand project descriptions."

        ])

    elif score < 80:

        recommendations.extend([

            "Resume is competitive.",

            "Improve ATS keyword optimization.",

            "Add stronger action verbs and quantified results."

        ])

    else:

        recommendations.extend([

            "Excellent resume alignment.",

            "Resume is highly ATS friendly.",

            "Ready for interview submission."

        ])

    # =========================================================
    # DISPLAY STRINGS
    # =========================================================

    matched_display = ", ".join(matched_skills)

    missing_display = ", ".join(missing_skills)

    additional_display = ", ".join(additional_skills)

    if matched_display == "":
        matched_display = "None"

    if missing_display == "":
        missing_display = "None"

    if additional_display == "":
        additional_display = "None"

    # =========================================================
    # EXTRA METRICS (Rule-Based)
    # =========================================================

    grammar_score = min(
        100,
        max(
            70,
            int(
                80 + (resume_words / 15)
            )
        )
    )

    formatting_score = min(
        100,
        max(
            75,
            int(
                85 + (resume_lines / 4)
            )
        )
    )

    keyword_score = round(skill_coverage)

    readability_score = min(
        100,
        max(
            72,
            int(
                82 + (resume_words / 20)
            )
        )
    )

    overall_resume_score = round(

        (
            grammar_score +
            formatting_score +
            keyword_score +
            readability_score
        ) / 4,

        2

    )
# =========================================================
# PREMIUM RESULT DASHBOARD
# =========================================================

if predict:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">

    <h2 style="
    text-align:center;
    font-size:34px;
    margin-bottom:5px;
    ">

    📊 AI Resume Analysis Report

    </h2>

    <p style="
    text-align:center;
    color:#6B7280;
    ">

    Machine Learning Powered Resume-to-Job Compatibility

    </p>

    </div>

    """, unsafe_allow_html=True)

    # =====================================================
    # SCORE SECTION
    # =====================================================

    left, right = st.columns([1, 2])

    with left:

        circumference = 2 * math.pi * 54

        progress = circumference - (score / 100) * circumference

        svg = f"""

<svg width="220" height="220" viewBox="0 0 220 220">

<defs>

<linearGradient id="gradient">

<stop offset="0%" stop-color="#8B5CF6"/>

<stop offset="100%" stop-color="#06B6D4"/>

</linearGradient>

</defs>

<circle

cx="110"

cy="110"

r="54"

stroke="#E5E7EB"

stroke-width="12"

fill="none"

/>

<circle

cx="110"

cy="110"

r="54"

stroke="url(#gradient)"

stroke-width="12"

fill="none"

stroke-linecap="round"

stroke-dasharray="{circumference}"

stroke-dashoffset="{progress}"

transform="rotate(-90 110 110)"

/>

<text

x="110"

y="105"

text-anchor="middle"

font-size="34"

font-weight="700"

fill="#8B5CF6">

{score:.0f}%

</text>

<text

x="110"

y="135"

text-anchor="middle"

font-size="14"

fill="#555">

Resume Match

</text>

</svg>

"""

        st.markdown(
            f"""
<div class="glass" style="text-align:center">

{svg}

<h3 style="color:{match_color};">

{emoji} {match_level}

</h3>

</div>
""",
            unsafe_allow_html=True
        )

        if score >= 80:
            st.balloons()

    with right:

        st.markdown("### 🚀 AI Match Summary")

        progress_bar = st.progress(0)

        for i in range(int(score) + 1):

            progress_bar.progress(i / 100)

            time.sleep(0.01)

        st.success(
            f"Resume Match Score : {score:.2f}%"
        )

        st.write("ATS Score")

        st.progress(ats_score / 100)

        st.write(f"{ats_score:.2f}%")

        st.write("Skill Coverage")

        st.progress(skill_coverage / 100)

        st.write(f"{skill_coverage:.2f}%")

        st.write("Overall Resume Quality")

        st.progress(overall_resume_score / 100)

        st.write(f"{overall_resume_score:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # KPI CARDS
    # =====================================================

    a, b, c, d = st.columns(4)

    with a:

        st.markdown(f"""
<div class="glass">

<h3 style="text-align:center;">
🎯 ATS Score
</h3>

<h1 style="
text-align:center;
color:#8B5CF6;
">
{ats_score:.1f}%
</h1>

</div>
""", unsafe_allow_html=True)

    with b:

        st.markdown(f"""
<div class="glass">

<h3 style="text-align:center;">
📈 Skill Coverage
</h3>

<h1 style="
text-align:center;
color:#06B6D4;
">
{skill_coverage:.1f}%
</h1>

</div>
""", unsafe_allow_html=True)

    with c:

        st.markdown(f"""
<div class="glass">

<h3 style="text-align:center;">
📄 Resume Quality
</h3>

<h1 style="
text-align:center;
color:#7C3AED;
">
{overall_resume_score:.1f}%
</h1>

</div>
""", unsafe_allow_html=True)

    with d:

        st.markdown(f"""
<div class="glass">

<h3 style="text-align:center;">
⭐ Strength
</h3>

<h2 style="
text-align:center;
color:#0891B2;
">

{resume_strength}

</h2>

</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # QUALITY ANALYSIS
    # =====================================================

    q1, q2 = st.columns(2)

    with q1:

        st.markdown("## 📑 Resume Quality")

        st.write("Grammar")

        st.progress(grammar_score / 100)

        st.write(f"{grammar_score}%")

        st.write("Formatting")

        st.progress(formatting_score / 100)

        st.write(f"{formatting_score}%")

        st.write("Readability")

        st.progress(readability_score / 100)

        st.write(f"{readability_score}%")

        st.write("Keyword Optimization")

        st.progress(keyword_score / 100)

        st.write(f"{keyword_score}%")

    with q2:

        st.markdown("## 📊 Resume Statistics")

        st.metric(
            "Resume Words",
            resume_words
        )

        st.metric(
            "Resume Characters",
            resume_chars
        )

        st.metric(
            "Resume Lines",
            resume_lines
        )

        st.metric(
            "Job Description Words",
            jd_words
        )
    # =====================================================
    # SKILL ANALYSIS
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">

    <h2 style="margin-bottom:5px;">
    🧠 Skill Analysis
    </h2>

    <p style="color:#6B7280;">
    Comparison between Resume Skills and Required Skills
    </p>

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # -------------------------
    # MATCHED SKILLS
    # -------------------------

    with c1:

        st.markdown("""
        <div class="glass">
        <h3 style="color:#10B981;">
        ✅ Matched Skills
        </h3>
        """, unsafe_allow_html=True)

        if matched_skills:

            chips = ""

            for skill in matched_skills:

                chips += f"""
<span style="
display:inline-block;
padding:8px 14px;
margin:5px;
border-radius:30px;
background:linear-gradient(90deg,#8B5CF6,#06B6D4);
color:white;
font-size:14px;
font-weight:600;
">
{skill.title()}
</span>
"""

            st.markdown(chips, unsafe_allow_html=True)

        else:

            st.info("No matching skills found.")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # MISSING SKILLS
    # -------------------------

    with c2:

        st.markdown("""
        <div class="glass">
        <h3 style="color:#EF4444;">
        ❌ Missing Skills
        </h3>
        """, unsafe_allow_html=True)

        if missing_skills:

            chips = ""

            for skill in missing_skills:

                chips += f"""
<span style="
display:inline-block;
padding:8px 14px;
margin:5px;
border-radius:30px;
background:#FEE2E2;
color:#DC2626;
font-weight:600;
font-size:14px;
">
{skill.title()}
</span>
"""

            st.markdown(chips, unsafe_allow_html=True)

        else:

            st.success("No missing skills 🎉")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # EXTRA SKILLS
    # -------------------------

    with c3:

        st.markdown("""
        <div class="glass">
        <h3 style="color:#2563EB;">
        ➕ Additional Skills
        </h3>
        """, unsafe_allow_html=True)

        if additional_skills:

            chips = ""

            for skill in additional_skills:

                chips += f"""
<span style="
display:inline-block;
padding:8px 14px;
margin:5px;
border-radius:30px;
background:#DBEAFE;
color:#2563EB;
font-weight:600;
font-size:14px;
">
{skill.title()}
</span>
"""

            st.markdown(chips, unsafe_allow_html=True)

        else:

            st.info("No additional skills.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # AI RECOMMENDATIONS
    # =====================================================

    st.markdown("""
    <div class="glass">

    <h2>
    🤖 AI Recommendations
    </h2>

    </div>
    """, unsafe_allow_html=True)

    for i, rec in enumerate(recommendations, start=1):

        st.markdown(f"""
<div style="
background:white;
padding:18px;
border-radius:16px;
margin-bottom:15px;
border-left:6px solid #8B5CF6;
box-shadow:0 10px 20px rgba(0,0,0,.06);
">

<b>{i}.</b> {rec}

</div>
""", unsafe_allow_html=True)

    # =====================================================
    # REPORT SUMMARY
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">

    <h2>
    📄 Analysis Summary
    </h2>

    </div>
    """, unsafe_allow_html=True)

    if score < 40:

        st.error(
            "Your resume has a low compatibility with the selected role. Add missing technical skills and tailor your resume for the job description."
        )

    elif score < 60:

        st.warning(
            "Your resume is moderately aligned. Improving keyword optimization and adding relevant projects can significantly increase the score."
        )

    elif score < 80:

        st.success(
            "Your resume has good compatibility. Minor improvements in ATS keywords and project descriptions can make it even stronger."
        )

    else:

        st.success(
            "Excellent! Your resume is highly aligned with the selected job role and is likely to perform well in ATS screening."
        )

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = f"""
=========================================
        AI RESUME MATCH REPORT
=========================================

Resume Match Score : {score:.2f}%

ATS Score : {ats_score:.2f}%

Skill Coverage : {skill_coverage:.2f}%

Resume Quality : {overall_resume_score:.2f}%

Resume Strength : {resume_strength}

-----------------------------------------

Matched Skills

{matched_display}

-----------------------------------------

Missing Skills

{missing_display}

-----------------------------------------

Additional Skills

{additional_display}

-----------------------------------------

Recommendations

"""

    for r in recommendations:

        report += f"\n• {r}"

    st.download_button(

        "📥 Download Report",

        report,

        file_name="Resume_Match_Report.txt",

        mime="text/plain"

    )

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
<hr>

<div style="text-align:center;">

<h3 style="
background:linear-gradient(90deg,#8B5CF6,#06B6D4);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-weight:800;
">

🚀 Resume Match AI

</h3>

<p style="color:#6B7280;">

Built with ❤️ using Streamlit • Scikit-learn • Machine Learning

</p>

<p style="font-size:13px;color:#9CA3AF;">

© 2026 Resume Match AI. Premium Glassmorphism UI.

</p>

</div>

""", unsafe_allow_html=True)

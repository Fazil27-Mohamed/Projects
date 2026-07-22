# Resume-to-Job Match Scorer

An AI-powered Resume-to-Job Match Scorer that predicts how well a candidate's resume matches a job description using Machine Learning and Natural Language Processing (NLP).

---

## Features

- Resume and Job Description Matching
- Skill Extraction and Comparison
- TF-IDF Vectorization
- Machine Learning-based Match Score Prediction
- Match Quality Classification
- Simple Web Interface

---

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Pickle
- HTML
- CSS

---

## Dataset

The dataset contains the following information:

- Resume Text
- Job Description
- Resume Skills
- Required Skills
- Job Category
- AI Match Score
- Skill String Match Score
- Fuzzy Match Score

---

## Machine Learning Workflow

- Data Collection
- Data Preprocessing
- Text Cleaning
- TF-IDF Feature Extraction
- Model Training
- Model Evaluation
- Resume Match Score Prediction

---

## Project Structure

```
resume-to-job-match/
│
├── models/
│   ├── resume_match_model.pkl
│   ├── resume_tfidf.pkl
│   ├── job_tfidf.pkl
│   ├── resume_skill_tfidf.pkl
│   ├── skill_tfidf.pkl
│   └── category_encoder.pkl
│
├── app.py
├── Resume_to_job_match.ipynb
├── requirements.txt
└── README.md
```

---

## Model

**Machine Learning Algorithm**

- Random Forest Regressor

**Saved Model Files**

- resume_match_model.pkl
- resume_tfidf.pkl
- job_tfidf.pkl
- resume_skill_tfidf.pkl
- skill_tfidf.pkl
- category_encoder.pkl

---

## Future Enhancements

- Semantic Resume Matching using BERT
- Resume Ranking System
- Explainable Match Score
- Multiple Resume Upload Support
- Enhanced Skill Recommendation

---

## Author

**Mohamed Fazil S**

Artificial Intelligence & Data Science Undergraduate

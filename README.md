
# Student Placement Predictor

An interactive machine-learning application that predicts
placement status from academic and experience-related features.

## Features
- Data preprocessing with scikit-learn pipelines
- Logistic Regression classification
- Train/test evaluation using accuracy and F1-score
- Interactive Streamlit prediction interface
- Excludes salary to avoid post-placement information leakage

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `placement.csv` in the project root before running.

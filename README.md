# PhishGuard — ML-Based Phishing Website Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![Accuracy](https://img.shields.io/badge/Accuracy-90.09%25-brightgreen)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)

## Project Overview
PhishGuard is a Machine Learning-based phishing website 
detection system that classifies URLs as phishing or 
legitimate using 15 URL-structural features.

## Team Members
| Name | Roll Number |
|------|-------------|
| Shaik Mahaboob Subhani | N210243 |
| Karthik Challagundla | N210013 |
| Kunda Venkata Sai Subhash | N210002 |

## Results
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 90.09% | 89.10% |
| SVM | 88.63% | 87.38% |
| Decision Tree | 87.90% | 86.31% |
| Logistic Regression | 84.87% | 83.37% |

## Dataset
- Source: GregaVrbancic Phishing Dataset
- Total URLs: 11,430
- After cleaning: 10,245
- Features selected: 15 out of 88

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Download dataset
python download_data.py

### 3. Train the model
python train.py

### 4. Launch web app
streamlit run app.py

## Tech Stack
- Python 3.12
- scikit-learn
- pandas, NumPy
- Streamlit
- Matplotlib, Seaborn
- joblib

## Project Structure
phishing_detector/
├── data/
├── src/
│   ├── feature_extraction.py
│   ├── data_processing.py
│   ├── model_training.py
│   ├── visualizations.py
│   └── prediction.py
├── models/
├── visualizations/
├── train.py
├── app.py
└── requirements.txt

## Summer Internship
Rgukt Nuzvid


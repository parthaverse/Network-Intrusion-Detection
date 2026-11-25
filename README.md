# 🛡️ Network Intrusion Detection System
## Project Overview
This project is a machine learning-based Network Intrusion Detection System that identifies normal and malicious network traffic using the NSL-KDD dataset. It compares three machine learning algorithms and selects the best-performing model for intrusion detection.
## Objectives

- Analyze network traffic and identify malicious activities.
- Perform data cleaning and exploratory data analysis.
- Train and compare machine learning classification models.
- Improve model performance through hyperparameter tuning.
- Develop an interactive Streamlit dashboard.

## Dataset

- Dataset: NSL-KDD (20% Training Subset)
- Original records: 25,192
- Input features: 41
- Classification: Binary Classification
- Normal Traffic: 0
- Malicious Traffic: 1

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Visual Studio Code

## Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Methodology

1. Dataset collection
2. Data cleaning
3. Exploratory data analysis
4. Feature encoding and scaling
5. Training and testing split (80:20)
6. Model training
7. 5-fold cross-validation
8. Hyperparameter tuning using GridSearchCV
9. Model evaluation and comparison
10. Best model selection
11. Streamlit dashboard development

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## Data Visualization

The project includes 15 visualizations covering network traffic distribution, attack types, network protocols, correlation analysis, model comparison, confusion matrices, cross-validation and ROC curve.

## Streamlit Dashboard

The interactive dashboard contains six sections:

- Home
- Dataset Overview
- Data Visualization
- Model Performance
- Intrusion Prediction
- About Project

## How to Run the Project

Install the required libraries:

`python -m pip install -r requirements.txt`

Perform data analysis:

`python data_analysis.py`

Train machine learning models:

`python train_models.py`

Launch the Streamlit dashboard:

`python -m streamlit run app.py`

## Project Structure

- dataset/ — Original and cleaned datasets
- graphs/ — Data analysis and model evaluation graphs
- models/ — Trained machine learning models
- results/ — Evaluation results and reports
- data_analysis.py — Data cleaning and visualization
- train_models.py — Model training and evaluation
- app.py — Interactive Streamlit dashboard
- requirements.txt — Required Python libraries
- README.md — Project documentation

## Conclusion

This project demonstrates how machine learning can be applied to network intrusion detection. It compares three classification algorithms and selects the best model based on cross-validation performance. The system is designed for academic purposes using historical benchmark data and is not a live network monitoring system.

## Author
Partha Biswas
MCA, Section: D
Student Code: BWU/MCA/25/209
Data Science Project

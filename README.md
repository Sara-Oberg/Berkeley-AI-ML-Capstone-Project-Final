### Berkeley-AI-ML-Capstone-Project-Final

### Project Title

### ❤️ Predicting Heart Disease Using Machine Learning

A machine learning project to explore and predict heart disease using patient health records. This project includes exploratory data analysis (EDA), data cleaning, feature engineering, and a baseline classification model to identify individuals at risk of heart disease.

**Author**

Sara Obergassel

#### Executive summary

This project explores whether heart disease can be predicted using personal and health-related data. I applied several machine learning models, tuned them using cross-validation, and evaluated them using performance metrics like recall and ROC AUC. The final model helps identify individuals at risk of heart disease.

#### Rationale

Heart disease is one of the leading causes of death worldwide. Being able to predict it early using basic health data could help save lives by enabling faster diagnosis and preventive care. This model could support healthcare professionals and increase public awareness.

#### Research Question

Can we predict heart disease in individuals using their basic personal and health data?

#### Data Sources

The dataset is from Kaggle:
Heart Disease Dataset – [Kaggle](https://www.kaggle.com/datasets/mirzahasnine/heart-disease-dataset)
It includes information such as age, blood pressure, cholesterol, glucose, smoking habits, and whether the individual had a heart disease or stroke.

#### Methodology
Data Cleaning & Preprocessing (missing values, outliers, encoding, scaling)
Feature Engineering
Model Training: Logistic Regression, Naive Bayes, SVM, Random Forest, KNN, XGBoost, and MLP
Hyperparameter Tuning: GridSearchCV
Evaluation: Accuracy, Recall, F1 Score, ROC AUC, Confusion Matrices, SHAP analysis

#### Project Workflow

The project follows a structured ML pipeline, including:

1. Import libraries
2. Data Loading
3. Exploratory Data
4. Evaluate Data Analyziz
5. Feature Engineering
6. Train/Test split
7. Regression Modeling
8. Other Models
9. Hyperparameter tuning
10. Selecting best model
11. Model interpretation
12. Results
    
#### Results

In this project, we aimed to predict the risk of heart disease or stroke using different machine learning models based on patient health data. We cleaned the data, handled missing values and outliers, and tested several models: Logistic Regression, SVM, Random Forest, K-Nearest Neighbors, XGBoost, and MLP.
After comparing their performance using accuracy, recall, F1 score, and ROC AUC, we selected Support Vector Machine (SVM) as the best model.

Why SVM?
Recall: 0.685, this means it correctly identified about 69% of true heart disease/stroke cases.
F1 Score: 0.326, this shows a balance between precision and recall.
ROC AUC: 0.722, this indicates the model is fairly good at distinguishing between positive and negative cases.
Other models like XGBoost and Random Forest had higher accuracy but much lower recall. This means they often missed actual cases, which is risky in health predictions. Therefore, SVM gives a better trade-off, especially when recall is more important.

Visual Results:
The confusion matrix showed that SVM correctly predicted most non-disease cases and a good number of true disease cases.
The ROC curve showed good separation from the random guess line.
The SHAP analysis revealed that age, systolic blood pressure, and cigarettes per day had the most influence on predictions.
The residual plot showed most predictions were accurate with few errors.

Conclusion:
SVM performed best for this classification task, especially because of its high recall. This model can be useful in real-world settings to support early detection of heart disease or stroke risk, which can help prevent serious outcomes.

#### Next steps

- Try SMOTE or other resampling methods to improve recall further.
  
- Add external clinical datasets to validate generalizability.
  
- Build a simple interactive dashboard or app for user-friendly predictions.

#### Outline of project

https://github.com/Sara-Oberg/Berkeley-AI-ML-Capstone-Project-Final/blob/main/Heart_Disease_Capstone_Project_Sara_Obergassel_24Mai2025_Final.ipynb

https://github.com/Sara-Oberg/Berkeley-AI-ML-Capstone-Project-Final/blob/main/heart_disease.csv

https://github.com/Sara-Oberg/Berkeley-AI-ML-Capstone-Project-Final/tree/main/data

https://github.com/Sara-Oberg/Berkeley-AI-ML-Capstone-Project-Final/tree/main/images


##### Contact and Further Information

For questions or collaboration opportunities, feel free to contact me at
pashmine (at) email.de

# Loan Approval Prediction System — README
# ==========================================

# 🏦 Loan Approval Prediction System

> An end-to-end Machine Learning project that predicts loan approval outcomes using applicant financial and demographic data, deployed as a professional Streamlit web application.

---

## 📌 Problem Statement

Financial institutions process thousands of loan applications daily. Manually reviewing each application is time-consuming and prone to human bias. This project automates the loan approval decision by training a machine learning model on 20,000 historical loan records, providing instant, data-driven approval predictions.

---

## 📂 Project Structure

```
loan_project/
│
├── data/
│   └── loan_dataset_20000.csv      # Raw dataset (20,000 records)
│
├── model/
│   ├── best_model.pkl              # Trained & tuned model (Random Forest)
│   ├── scaler.pkl                  # StandardScaler for numerical features
│   ├── feature_columns.pkl         # Ordered list of encoded feature columns
│   ├── model_metadata.json         # Model performance metrics & best params
│   └── plots/                      # EDA & evaluation visualizations
│       ├── target_distribution.png
│       ├── credit_score_dist.png
│       ├── dti_dist.png
│       ├── correlation_heatmap.png
│       ├── feature_importance.png
│       ├── cm_logistic_regression.png
│       └── cm_random_forest.png
│
├── app.py                          # Streamlit web application
├── train.py                        # ML training pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 📊 Dataset

| Property      | Details                         |
|---------------|---------------------------------|
| Records       | 20,000 applicants               |
| Features      | 21 columns                      |
| Target        | `loan_status` (Approved/Rejected) |
| Train/Test    | 80% / 20% split (stratified)    |

### Key Features Used

| Feature                  | Type        | Description                         |
|--------------------------|-------------|-------------------------------------|
| `age`                    | Numerical   | Applicant age                       |
| `annual_income`          | Numerical   | Yearly income                       |
| `monthly_income`         | Numerical   | Monthly income                      |
| `credit_score`           | Numerical   | Credit score (300–900)              |
| `debt_to_income_ratio`   | Numerical   | Debt as fraction of income          |
| `loan_amount`            | Numerical   | Requested loan amount               |
| `interest_rate`          | Numerical   | Interest rate (%)                   |
| `loan_term`              | Numerical   | Loan duration (months)              |
| `gender`                 | Categorical | Male / Female                       |
| `marital_status`         | Categorical | Single / Married / Divorced         |
| `education_level`        | Categorical | High School / Bachelor's / Master's / Ph.D. |
| `employment_status`      | Categorical | Employed / Self-Employed / etc.     |
| `loan_purpose`           | Categorical | Car / Home / Business / etc.        |

---

## 🎯 Target Variable Creation

The dataset does **not** include a pre-labeled target. The `loan_status` column is engineered using domain rules:

```python
if credit_score >= 750 AND debt_to_income_ratio < 0.20:
    loan_status = "Approved"
elif credit_score >= 650 AND debt_to_income_ratio < 0.35:
    loan_status = "Approved"
else:
    loan_status = "Rejected"
```

---

## ⚙️ ML Pipeline

1. **Data Loading** — Read CSV from `data/` directory
2. **Target Creation** — Rule-based `loan_status` column
3. **Preprocessing**
   - Missing value imputation (median for numerical, mode for categorical)
   - One-Hot Encoding for categorical columns
   - StandardScaler for numerical features
4. **EDA** — Distribution plots, heatmaps, and feature analysis saved as PNG
5. **Model Training**
   - Logistic Regression
   - Random Forest Classifier
6. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix, Cross-Validation
7. **Hyperparameter Tuning** — GridSearchCV on Random Forest
8. **Model Saving** — Pickle + JSON metadata

---

## 🤖 Model Performance

| Model                        | Accuracy | F1-Score |
|------------------------------|----------|----------|
| Logistic Regression          | ~XX%     | ~X.XXXX  |
| Random Forest (Tuned)        | ~XX%     | ~X.XXXX  |

> *Actual values written to `model/model_metadata.json` after running `train.py`*

---

## 🚀 How to Run

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model

```bash
cd loan_project
python train.py
```

This will:
- Create and save the trained model to `model/best_model.pkl`
- Save the scaler to `model/scaler.pkl`
- Generate EDA plots to `model/plots/`
- Print a full evaluation report

### Step 3 — Launch the Web App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🌐 Web App Features

| Feature               | Description                                    |
|-----------------------|------------------------------------------------|
| 🏠 Predict Loan       | Full input form + instant prediction           |
| 📊 Model Insights     | EDA plots + performance metrics in tabbed view |
| ℹ️ About              | Project info, tech stack, and usage guide      |
| 🎨 Dark UI            | Glassmorphism design with gradient themes      |
| 📊 Gauge Chart        | Plotly-powered approval probability meter      |
| 💡 Improvement Tips   | Actionable suggestions on rejection            |

---

## 📸 Screenshots

> *(Run the app and take screenshots to add here)*

| Prediction Form | Result — Approved | Result — Rejected |
|:-:|:-:|:-:|
| ![form](screenshots/form.png) | ![approved](screenshots/approved.png) | ![rejected](screenshots/rejected.png) |

---

## 🧰 Tech Stack

| Layer            | Tool/Library              |
|------------------|---------------------------|
| Language         | Python 3.9+               |
| ML               | Scikit-learn              |
| Web App          | Streamlit                 |
| Visualization    | Plotly, Seaborn, Matplotlib |
| Data Processing  | Pandas, NumPy             |
| Model Saving     | Pickle (built-in)         |

---

## 👤 Author

**Loan AI Project** — Built as a beginner-friendly yet professional ML project for learning end-to-end machine learning workflows.

---

## 📄 License

This project is open-source and available for educational use.

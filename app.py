# =============================================================================
# Loan Approval Prediction System - Streamlit Web App
# =============================================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — Premium Dark Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0e0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(8px);
}

/* ── Section Header ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #a78bfa;
    border-bottom: 2px solid rgba(167,139,250,0.4);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Hero Card ── */
.hero-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(59,130,246,0.25));
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin-bottom: 2rem;
}
.hero-card h1 { font-size: 2.4rem; font-weight: 700; color: #fff; margin-bottom: 0.3rem; }
.hero-card p  { font-size: 1.05rem; color: #c4b5fd; }

/* ── Input Labels ── */
label { color: #c4b5fd !important; font-weight: 500; }

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.6) !important;
}

/* ── Result Boxes ── */
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(5,150,105,0.15));
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: fadeIn 0.6s ease;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(185,28,28,0.15));
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: fadeIn 0.6s ease;
}
.result-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-sub   { font-size: 1rem; color: #cbd5e1; }

/* ── Probability Bar ── */
.prob-bar-wrap {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    height: 14px;
    margin: 0.6rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}

/* ── Glassmorphism Cards ── */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1.5rem;
    backdrop-filter: blur(8px);
    margin-bottom: 1rem;
}

/* ── Fade-in ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 8px 8px 0 0;
    color: #c4b5fd;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(124,58,237,0.35) !important;
    color: #fff !important;
    border-bottom: 2px solid #7c3aed;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Select / Input boxes ── */
.stSelectbox > div, .stNumberInput > div > div, .stSlider {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# ---------------------------------------------------------------------------
MODEL_DIR = "model"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler, and feature columns from disk."""
    model_path   = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    cols_path    = os.path.join(MODEL_DIR, "feature_columns.pkl")
    meta_path    = os.path.join(MODEL_DIR, "model_metadata.json")

    if not os.path.exists(model_path):
        return None, None, None, None

    with open(model_path,  "rb") as f: model   = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler  = pickle.load(f)
    with open(cols_path,   "rb") as f: feat_cols = pickle.load(f)
    with open(meta_path,   "r")  as f: metadata  = json.load(f)

    return model, scaler, feat_cols, metadata

model, scaler, feat_cols, metadata = load_artifacts()

# ---------------------------------------------------------------------------
# HELPER — Build encoded input row
# ---------------------------------------------------------------------------
def build_input_df(user_input: dict, feat_cols: list) -> pd.DataFrame:
    """
    Convert the user input dict into a DataFrame that matches the
    one-hot encoded feature space used during training.
    """
    raw = pd.DataFrame([user_input])

    # Categorical columns to one-hot encode (must match train.py)
    cat_cols = ['gender', 'marital_status', 'education_level',
                'employment_status', 'loan_purpose']

    raw_encoded = pd.get_dummies(raw, columns=cat_cols, drop_first=True)

    # Align with training feature columns
    final = raw_encoded.reindex(columns=feat_cols, fill_value=0)
    return final

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <h1>🏦 Loan Approval Prediction System</h1>
    <p>AI-powered loan eligibility assessment using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR — Navigation & Model Info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Predict Loan", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    if metadata:
        st.markdown("### 🤖 Model Info")
        st.markdown(f"**Model:** {metadata.get('model_name','N/A')}")
        st.markdown(f"**Accuracy:** `{metadata.get('accuracy',0)*100:.2f}%`")
        st.markdown(f"**F1-Score:** `{metadata.get('f1_score',0):.4f}`")
        params = metadata.get('best_params', {})
        st.markdown("**Best Params:**")
        for k, v in params.items():
            st.markdown(f"&nbsp;&nbsp;• `{k}`: {v}")
    else:
        st.warning("⚠️ Model not found.\nPlease run `train.py` first.")

    st.markdown("---")
    st.markdown("### 📋 Approval Rules")
    st.markdown("""
    ✅ **Approved** if:
    - Credit ≥ 750 & DTI < 0.20
    - Credit ≥ 650 & DTI < 0.35

    ❌ **Rejected** otherwise
    """)

# ===========================================================================
# PAGE 1: PREDICT LOAN
# ===========================================================================
if "🏠 Predict Loan" in page:

    if model is None:
        st.error("❌ Model artifacts not found! Please run `python train.py` first.")
        st.stop()

    st.markdown('<div class="section-header">📝 Applicant Information</div>', unsafe_allow_html=True)

    with st.form("loan_form"):
        # ── Row 1: Personal Details ──────────────────────────────────────
        st.markdown("#### 👤 Personal Details")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c3:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with c4:
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Bachelor's", "Master's", "Ph.D."]
            )

        # ── Row 2: Financial Details ─────────────────────────────────────
        st.markdown("#### 💰 Financial Details")
        c5, c6, c7 = st.columns(3)
        with c5:
            annual_income = st.number_input(
                "Annual Income (₹)", min_value=10000.0, max_value=500000.0,
                value=60000.0, step=1000.0, format="%.2f"
            )
        with c6:
            monthly_income = st.number_input(
                "Monthly Income (₹)", min_value=500.0, max_value=50000.0,
                value=annual_income / 12, step=100.0, format="%.2f"
            )
        with c7:
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Self-Employed", "Unemployed", "Retired"]
            )

        # ── Row 3: Loan & Credit Details ─────────────────────────────────
        st.markdown("#### 🏦 Loan & Credit Details")
        c8, c9, c10 = st.columns(3)
        with c8:
            credit_score = st.slider(
                "Credit Score", min_value=300, max_value=900, value=700, step=10
            )
        with c9:
            debt_to_income_ratio = st.slider(
                "Debt-to-Income Ratio", min_value=0.0, max_value=1.0,
                value=0.25, step=0.01, format="%.2f"
            )
        with c10:
            loan_amount = st.number_input(
                "Loan Amount (₹)", min_value=1000.0, max_value=500000.0,
                value=25000.0, step=500.0, format="%.2f"
            )

        c11, c12, c13 = st.columns(3)
        with c11:
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["Car", "Home", "Business", "Education", "Debt consolidation",
                 "Medical", "Personal", "Other"]
            )
        with c12:
            interest_rate = st.number_input(
                "Interest Rate (%)", min_value=1.0, max_value=30.0,
                value=10.0, step=0.5, format="%.1f"
            )
        with c13:
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

        c14, c15, c16 = st.columns(3)
        with c14:
            installment = st.number_input(
                "Monthly Installment (₹)", min_value=10.0, max_value=20000.0,
                value=500.0, step=50.0, format="%.2f"
            )
        with c15:
            num_of_open_accounts = st.number_input(
                "Open Accounts", min_value=0, max_value=30, value=5, step=1
            )
        with c16:
            total_credit_limit = st.number_input(
                "Total Credit Limit (₹)", min_value=0.0, max_value=500000.0,
                value=80000.0, step=1000.0, format="%.2f"
            )

        c17, c18, c19 = st.columns(3)
        with c17:
            current_balance = st.number_input(
                "Current Balance (₹)", min_value=0.0, max_value=500000.0,
                value=15000.0, step=500.0, format="%.2f"
            )
        with c18:
            delinquency_history = st.number_input(
                "Delinquency History", min_value=0, max_value=10, value=0, step=1
            )
        with c19:
            num_of_delinquencies = st.number_input(
                "# Delinquencies", min_value=0, max_value=20, value=0, step=1
            )

        public_records = st.number_input(
            "Public Records", min_value=0, max_value=10, value=0, step=1
        )

        st.markdown("")
        submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

    # ── PREDICTION ─────────────────────────────────────────────────────────
    if submitted:
        user_input = {
            'age'                  : age,
            'annual_income'        : annual_income,
            'monthly_income'       : monthly_income,
            'debt_to_income_ratio' : debt_to_income_ratio,
            'credit_score'         : credit_score,
            'loan_amount'          : loan_amount,
            'interest_rate'        : interest_rate,
            'loan_term'            : loan_term,
            'installment'          : installment,
            'num_of_open_accounts' : num_of_open_accounts,
            'total_credit_limit'   : total_credit_limit,
            'current_balance'      : current_balance,
            'delinquency_history'  : delinquency_history,
            'public_records'       : public_records,
            'num_of_delinquencies' : num_of_delinquencies,
            'gender'               : gender,
            'marital_status'       : marital_status,
            'education_level'      : education_level,
            'employment_status'    : employment_status,
            'loan_purpose'         : loan_purpose,
        }

        try:
            input_df = build_input_df(user_input, feat_cols)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            prob_approved = probability[1]  # class 1 = Approved
            prob_rejected = probability[0]

            st.markdown("---")
            st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-approved">
                        <div class="result-title">✅ LOAN APPROVED</div>
                        <div class="result-sub">Congratulations! Your loan application has been approved.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-rejected">
                        <div class="result-title">❌ LOAN REJECTED</div>
                        <div class="result-sub">Unfortunately, your loan application has been rejected.</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")

                # Probability Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_approved * 100,
                    title={'text': "Approval Probability (%)", 'font': {'color': '#c4b5fd', 'size': 15}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#c4b5fd'},
                        'bar': {'color': "#10b981" if prob_approved >= 0.5 else "#ef4444"},
                        'bgcolor': 'rgba(255,255,255,0.05)',
                        'steps': [
                            {'range': [0,  40],  'color': 'rgba(239,68,68,0.2)'},
                            {'range': [40, 65],  'color': 'rgba(234,179,8,0.2)'},
                            {'range': [65, 100], 'color': 'rgba(16,185,129,0.2)'},
                        ],
                        'threshold': {
                            'line': {'color': '#a78bfa', 'width': 3},
                            'thickness': 0.75, 'value': 50
                        }
                    },
                    number={'suffix': '%', 'font': {'color': '#fff', 'size': 28}}
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e0e0e0'},
                    height=260,
                    margin=dict(t=40, b=10, l=30, r=30)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                # Probability bars
                st.markdown("#### 📊 Probability Breakdown")
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:#10b981; font-weight:600;">✅ Approved</span>
                        <span style="color:#10b981; font-weight:700;">{prob_approved*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill" style="width:{prob_approved*100:.1f}%;
                             background: linear-gradient(90deg,#10b981,#34d399);"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:1rem; margin-bottom:4px;">
                        <span style="color:#ef4444; font-weight:600;">❌ Rejected</span>
                        <span style="color:#ef4444; font-weight:700;">{prob_rejected*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill" style="width:{prob_rejected*100:.1f}%;
                             background: linear-gradient(90deg,#ef4444,#f87171);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📋 Key Inputs Summary")
                summary_data = {
                    "Field": ["Credit Score", "DTI Ratio", "Annual Income",
                               "Loan Amount", "Employment"],
                    "Value": [str(credit_score), f"{debt_to_income_ratio:.2f}",
                               f"₹{annual_income:,.0f}", f"₹{loan_amount:,.0f}",
                               employment_status]
                }
                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True
                )

                # Eligibility tip
                if prediction == 0:
                    st.markdown("""
                    <div class="glass-card" style="border-color:rgba(234,179,8,0.4);">
                        <p style="color:#fbbf24; font-weight:600; margin:0 0 0.5rem;">💡 Improvement Tips</p>
                        <ul style="color:#d1d5db; margin:0; padding-left:1.2rem;">
                            <li>Improve your credit score above 650</li>
                            <li>Reduce debt-to-income ratio below 0.35</li>
                            <li>Consider a smaller loan amount</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="glass-card" style="border-color:rgba(16,185,129,0.4);">
                        <p style="color:#10b981; font-weight:600; margin:0 0 0.5rem;">🎉 Great Profile!</p>
                        <p style="color:#d1d5db; margin:0;">Your financial profile meets our approval criteria. Review interest rates carefully before proceeding.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)

# ===========================================================================
# PAGE 2: MODEL INSIGHTS
# ===========================================================================
elif "📊 Model Insights" in page:
    st.markdown('<div class="section-header">📊 Model Insights & EDA</div>', unsafe_allow_html=True)

    PLOTS = os.path.join(MODEL_DIR, "plots")
    plot_files = {
        "Target Distribution"       : "target_distribution.png",
        "Credit Score Distribution" : "credit_score_dist.png",
        "Debt-to-Income Ratio"      : "dti_dist.png",
        "Correlation Heatmap"       : "correlation_heatmap.png",
        "Feature Importance"        : "feature_importance.png",
        "Confusion Matrix – LR"     : "cm_logistic_regression.png",
        "Confusion Matrix – RF"     : "cm_random_forest.png",
    }

    missing = [v for v in plot_files.values() if not os.path.exists(os.path.join(PLOTS, v))]
    if missing:
        st.warning("⚠️ Some plots are missing. Please run `python train.py` first to generate them.")

    tabs = st.tabs(list(plot_files.keys()))
    for tab, (title, fname) in zip(tabs, plot_files.items()):
        with tab:
            path = os.path.join(PLOTS, fname)
            if os.path.exists(path):
                st.image(path, use_container_width=True, caption=title)
            else:
                st.info(f"Plot not yet generated: `{fname}`")

    # Model comparison table
    if metadata:
        st.markdown("---")
        st.markdown("#### 🏆 Model Performance Summary")
        perf_df = pd.DataFrame([{
            "Model"    : metadata.get("model_name", "Random Forest"),
            "Accuracy" : f"{metadata.get('accuracy', 0)*100:.2f}%",
            "F1-Score" : f"{metadata.get('f1_score', 0):.4f}",
            "Best Params": str(metadata.get("best_params", {}))
        }])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE 3: ABOUT
# ===========================================================================
elif "ℹ️ About" in page:
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#a78bfa; margin-top:0;">🏦 Loan Approval Prediction System</h3>
        <p>This end-to-end machine learning project predicts loan approval decisions using applicant 
        financial and demographic data. The system uses a <strong>tuned Random Forest Classifier</strong> 
        trained on 20,000 loan records.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#60a5fa; margin-top:0;">🔧 Tech Stack</h4>
            <ul style="color:#d1d5db;">
                <li><strong>Language:</strong> Python 3.9+</li>
                <li><strong>ML Library:</strong> Scikit-learn</li>
                <li><strong>App Framework:</strong> Streamlit</li>
                <li><strong>Visualization:</strong> Plotly, Seaborn, Matplotlib</li>
                <li><strong>Data:</strong> Pandas, NumPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#60a5fa; margin-top:0;">📊 Dataset</h4>
            <ul style="color:#d1d5db;">
                <li><strong>Records:</strong> 20,000 applicants</li>
                <li><strong>Features:</strong> 21 columns</li>
                <li><strong>Target:</strong> loan_status (Approved/Rejected)</li>
                <li><strong>Split:</strong> 80% train / 20% test</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#60a5fa; margin-top:0;">🤖 ML Pipeline</h4>
            <ol style="color:#d1d5db;">
                <li>Target variable creation (rule-based)</li>
                <li>Missing value imputation</li>
                <li>One-Hot Encoding (categorical)</li>
                <li>StandardScaler (numerical)</li>
                <li>Model training (LR + Random Forest)</li>
                <li>GridSearchCV hyperparameter tuning</li>
                <li>5-fold cross-validation</li>
                <li>Model serialization (pickle)</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#60a5fa; margin-top:0;">🚀 How to Run</h4>
            <ol style="color:#d1d5db;">
                <li>Install: <code>pip install -r requirements.txt</code></li>
                <li>Train: <code>python train.py</code></li>
                <li>Launch: <code>streamlit run app.py</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="text-align:center; margin-top:1rem;">
        <p style="color:#6b7280; margin:0;">
            Built with ❤️ | Loan Approval Prediction System | ML Project
        </p>
    </div>
    """, unsafe_allow_html=True)

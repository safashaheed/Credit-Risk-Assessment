
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. LOAD MODEL AND PREPROCESSOR
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "xgb_model.pkl")
)

preprocessor = joblib.load(
    os.path.join(BASE_DIR, "models", "preprocessor.pkl")
)

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CreditWise | Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

header {
    background-color: transparent;
}

h1, h2, h3 {
    color: #17365d;
}

p, label {
    color: #344563;
}

/* Header banner */
.hero {
    background: linear-gradient(120deg, #142d4e, #245b91);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    color: #e2ebf7;
    font-size: 16px;
    margin-bottom: 0;
}


/* Result dashboard cards */
.result-card {
    background: #ffffff;
    padding: 25px 22px;
    border-radius: 14px;
    border: 1px solid #dce5ef;
    box-shadow: 0 4px 14px rgba(20, 45, 78, 0.07);
    min-height: 145px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.result-label {
    color: #65758b;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.result-value {
    color: #17365d;
    font-size: 30px;
    font-weight: 750;
    margin-top: 14px;
}

.risk-low {
    color: #16834a;
}

.risk-medium {
    color: #c18416;
}

.risk-high {
    color: #c0392b;
}



/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #17365d !important;
    color: #ffffff !important;
    border-radius: 9px;
    border: none;
    height: 48px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #245b91 !important;
    color: #ffffff !important;
    border: none;
}

.stButton > button p,
.stFormSubmitButton > button p {
    color: #ffffff !important;
}


/* Input containers */
div[data-testid="stExpander"] {
    background-color: white;
    border-radius: 10px;
    border: 1px solid #e1e8f0;
}

div[data-testid="stForm"] {
    background-color: white;
    padding: 22px;
    border-radius: 13px;
    border: 1px solid #e1e8f0;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 16px;
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #7a8798;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🏦 CreditWise")

    st.caption("Machine Learning Credit Risk Assessment")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "An intelligent system that estimates the likelihood "
        "of an applicant experiencing payment difficulties."
    )

    st.divider()

    st.markdown("### Risk Categories")

    st.success("Low Risk: Below 30%")

    st.warning("Medium Risk: 30% to below 60%")

    st.error("High Risk: 60% or above")

    st.divider()

    st.caption("Academic Machine Learning Project")

# ==========================================
# 5. HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🏦 Credit Risk Assessment Dashboard</h1>
    <p>
        Intelligent loan applicant assessment powered by
        Machine Learning and XGBoost.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Evaluate applicant information and generate a model-based "
    "credit risk assessment."
)

st.write("")

# ==========================================
# 6. FEATURE INFORMATION
# ==========================================

num_features = list(preprocessor.transformers_[0][2])
cat_features = list(preprocessor.transformers_[1][2])

num_imputer = (
    preprocessor.named_transformers_["num"]
    .named_steps["imputer"]
)

cat_imputer = (
    preprocessor.named_transformers_["cat"]
    .named_steps["imputer"]
)

encoder = (
    preprocessor.named_transformers_["cat"]
    .named_steps["encoder"]
)

# ==========================================
# 7. APPLICANT FEATURE GROUPS
# ==========================================

personal_features = [
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "AGE"
]

employment_features = [
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "DAYS_EMPLOYED"
]

financial_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE"
]

other_features = [
    f for f in num_features + cat_features
    if f not in (
        personal_features
        + employment_features
        + financial_features
    )
]

# ==========================================
# 8. INPUT FUNCTION
# ==========================================

input_data = {}

def add_features(features):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            label = feature.replace("_", " ").title()

            if feature in num_features:

                index = num_features.index(feature)

                default_value = float(
                    num_imputer.statistics_[index]
                )

                input_data[feature] = st.number_input(
                    label,
                    value=default_value,
                    key=f"input_{feature}"
                )

            elif feature in cat_features:

                index = cat_features.index(feature)

                options = [
                    str(value)
                    for value in encoder.categories_[index]
                ]

                default_value = str(
                    cat_imputer.statistics_[index]
                )

                selected_index = (
                    options.index(default_value)
                    if default_value in options
                    else 0
                )

                input_data[feature] = st.selectbox(
                    label,
                    options=options,
                    index=selected_index,
                    key=f"input_{feature}"
                )

# ==========================================
# 9. APPLICANT FORM
# ==========================================

st.markdown("## 👤 Applicant Information")

st.caption(
    "Enter the applicant's details in the sections below."
)

with st.form("applicant_form"):

    st.markdown("### 1. Personal Information")

    add_features(personal_features)

    st.divider()

    st.markdown("### 2. Employment Information")

    add_features(employment_features)

    st.divider()

    st.markdown("### 3. Financial Information")

    add_features(financial_features)

    st.divider()

    with st.expander("➕ Additional Applicant Information"):

        st.caption(
            "Enter the remaining applicant details used by the model."
        )

        add_features(other_features)

    st.write("")

    submitted = st.form_submit_button(
        "🔍 Assess Credit Risk",
        use_container_width=True
    )

# ==========================================
# 10. PREDICTION
# ==========================================

if submitted:

    try:

        applicant = pd.DataFrame([input_data])

        applicant_processed = preprocessor.transform(applicant)

        probability = model.predict_proba(
            applicant_processed
        )[0][1]

        risk_score = 100 * (1 - probability)

        if probability < 0.30:

            risk_category = "Low Risk"
            risk_class = "risk-low"
            risk_icon = "🟢"

        elif probability < 0.60:

            risk_category = "Medium Risk"
            risk_class = "risk-medium"
            risk_icon = "🟠"

        else:

            risk_category = "High Risk"
            risk_class = "risk-high"
            risk_icon = "🔴"

        # ==================================
        # 11. RESULTS DASHBOARD
        # ==================================

        st.divider()

        st.markdown("## 📊 Credit Risk Assessment Results")

        st.caption(
            "The following results are generated by the trained XGBoost model."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">
                    Payment Difficulty Probability
                </div>
                <div class="result-value">
                    {probability * 100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">
                    Custom Risk Score
                </div>
                <div class="result-value">
                    {risk_score:.3f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">
                    Risk Classification
                </div>
                <div class="result-value {risk_class}">
                    {risk_icon} {risk_category}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Probability indicator

        st.markdown("### Risk Probability Indicator")

        st.progress(float(probability))

        st.caption(
            "Higher probability indicates a higher model-estimated "
            "likelihood of recorded payment difficulties."
        )

        # Interpretation

        st.markdown("### Assessment Summary")

        if probability < 0.30:

            st.success(
                "The model places this applicant in the Low Risk "
                "category based on the information entered."
            )

        elif probability < 0.60:

            st.warning(
                "The model places this applicant in the Medium Risk "
                "category. Further review may be appropriate."
            )

        else:

            st.error(
                "The model places this applicant in the High Risk "
                "category. Further review is recommended."
            )

        # Disclaimer

        st.info("""
        **Important Disclaimer**

        This is an academic machine-learning decision-support system.
        The model estimates the likelihood of recorded payment difficulties
        based on historical data.

        The custom risk score is not an official credit score.
        Predictions should not be used alone to approve or reject
        real loan applications.
        """)

    except Exception as e:

        st.error(f"Unable to generate assessment: {e}")

# ==========================================
# 12. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    CreditWise | Credit Risk Assessment & Loan Default Prediction
    <br>
    Academic Machine Learning Project
</div>
""", unsafe_allow_html=True)

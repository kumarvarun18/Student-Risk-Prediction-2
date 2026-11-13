from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# Page configuration  (must be the FIRST Streamlit command)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Risk Prediction",
    page_icon="🎓",
    layout="wide",
)

# Files are looked up next to this script first, then in common sub-folders,
# so the app works whether the repo is flat or organised in folders.
BASE_DIR = Path(__file__).parent


def find_file(pattern):
    for folder in [BASE_DIR, BASE_DIR / "assets", BASE_DIR / "model", BASE_DIR / "data"]:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    st.error(
        f"Required file '{pattern}' was not found in the repository. "
        "Upload it to the same folder as streamlit_app.py and redeploy."
    )
    st.stop()


CSS_PATH = find_file("style.css")
MODEL_PATH = find_file("student_risk_model.pkl")
PREPROCESSOR_PATH = find_file("preprocessor.pkl")
DATA_PATH = find_file("risk_students*.csv")

st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# Verified results of the final tuned model (20% hold-out test set, 400 students)
TEST_ACCURACY = 0.8075
TEST_PRECISION = 0.72   # class "At Risk"
TEST_RECALL = 0.54      # class "At Risk"
TEST_F1 = 0.62          # class "At Risk"
TEST_AUC = 0.8216
BASELINE_ACCURACY = 0.7125   # always predicting "Not At Risk"
CONFUSION = [[261, 24], [53, 62]]   # rows = actual, cols = predicted


# ---------------------------------------------------------------
# Cached loaders (load once, not on every click)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, preprocessor = load_artifacts()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Page",
    ["Home", "Dashboard", "Prediction", "Model Performance", "About"],
)

# ---------------------------------------------------------------
# Home
# ---------------------------------------------------------------
if menu == "Home":
    st.markdown(
        """
        <div class="hero">
        <h1 style="color:white">🎓 Student Performance Risk Prediction System</h1>
        <p>A Machine Learning web application for early identification of students who may be at academic risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        """
        <div class="custom-card">
            <h3>Project Overview</h3>
            <p>
            The Student Performance Risk Prediction System is a Machine Learning application that predicts
            whether a student is academically at risk based on academic and behavioural attributes such as
            GPA, attendance, assignment submission, lab performance, online learning hours, forum
            participation, and socio-economic status.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Project Objectives")
    col1, col2 = st.columns(2)
    with col1:
        st.success(
            """
            - Predict academic risk
            - Enable early intervention
            - Improve student success
            - Support educators"""
        )
    with col2:
        st.info(
            """
            - Analyze student data
            - Machine Learning Prediction
            - Interactive Dashboard
            - Streamlit Web Application"""
        )
    st.markdown("---")

    st.subheader("Technology Stack")
    tech1, tech2, tech3, tech4 = st.columns(4)
    tech1.metric("Language", "Python")
    tech2.metric("Framework", "Streamlit")
    tech3.metric("ML Library", "Scikit-learn")
    tech4.metric("Visualization", "Plotly")
    st.markdown("---")

    st.subheader("Project Workflow")
    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">
        <div class="workflow-card">Dataset</div>
        <div class="workflow-card">EDA</div>
        <div class="workflow-card">Preprocessing</div>
        <div class="workflow-card">Model Training</div>
        <div class="workflow-card">Hyperparameter Tuning</div>
        <div class="workflow-card">Deployment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("Key Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success(
            """
            - Interactive Dashboard
            - Data Visualization
            - Dataset Insights"""
        )
    with c2:
        st.success(
            """
            - Risk Prediction
            - Probability Score
            - Recommendations"""
        )
    with c3:
        st.success(
            """
            - Model Performance
            - Feature Importance
            - Project Information"""
        )
    st.markdown("---")

    st.subheader("Project Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Input Features", "10")
    s2.metric("Target", "Binary")
    s3.metric("Algorithm", "Gradient Boosting")
    s4.metric("Test Accuracy", f"{TEST_ACCURACY * 100:.2f}%")
    st.markdown("---")

    st.markdown(
        """
        <div class="highlight-box">
        <h2>Why This Project?</h2>
        <p>
            Educational institutions often face challenges in identifying students who require additional
            academic support. This application uses Machine Learning to analyze student information and
            provide early predictions, helping educators make informed decisions and improve student outcomes.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.success(
        "Navigate through the sidebar to explore the Dashboard, Prediction, Model Performance, and About pages."
    )

# ---------------------------------------------------------------
# About
# ---------------------------------------------------------------
elif menu == "About":
    st.header("About Project")
    st.markdown(
        """
    ## 🎓 Student Performance Risk Prediction System

    This project is a **Machine Learning-based Binary Classification System**
    developed to identify students who may be academically at risk.
    It analyzes academic and behavioural factors and predicts whether a student
    requires early academic intervention.

    The application was developed using **Python**, **Scikit-learn**, and
    **Streamlit** as part of a **45-Day Industrial Training Program**.
    """
    )

    st.divider()
    st.subheader("Features Used")
    features = [
        "Age",
        "Gender",
        "Socio Economic Status",
        "GPA",
        "Attendance Rate",
        "Assignments Submitted",
        "Online Learning Hours",
        "Forum Participation",
        "Lab Performance",
        "Extracurricular Participation",
    ]
    for feature in features:
        st.markdown(f"✅ {feature}")

    st.subheader("Project Objectives")
    st.markdown(
        """
    - Predict students who are academically at risk.
    - Support early intervention using data-driven insights.
    - Provide an interactive dashboard for data exploration.
    - Deploy the trained machine learning model using Streamlit.
    """
    )
    st.divider()

    st.subheader("Technologies Used")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Language", "Python")
    col2.metric("Framework", "Streamlit")
    col3.metric("ML Model", "Gradient Boosting")
    col4.metric("Library", "Scikit-learn")
    st.divider()

    st.header("Features")
    st.success(
        """
    - Interactive Dashboard
    - Dataset Filtering
    - Student Risk Prediction
    - Probability Score
    - Model Performance
    - Responsive User Interface
    """
    )
    st.divider()

    st.subheader("Project Workflow")
    st.markdown(
        "**Dataset → Data Exploration → EDA → Data Preprocessing → Model Training → "
        "Hyperparameter Tuning → Model Evaluation → Streamlit Deployment**"
    )
    st.divider()

    st.subheader("Machine Learning Details")
    st.markdown(
        """
    - **Learning Type:** Supervised Learning
    - **Problem Type:** Binary Classification
    - **Final Model:** Gradient Boosting Classifier (tuned with RandomizedSearchCV)
    - **Target Variable:** At_Risk
    """
    )
    st.divider()

    st.subheader("Future Enhancements")
    st.markdown(
        """
    - Database integration
    - User authentication
    - Real-time student monitoring
    - Downloadable prediction reports
    - Personalized academic recommendations
    """
    )
    st.divider()
    st.info(
        "This project demonstrates how Machine Learning can support educators "
        "in identifying academically at-risk students and enabling timely intervention."
    )

# ---------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------
elif menu == "Dashboard":
    st.header("Student Dataset Dashboard")
    df = load_data()
    risk_labels = {0: "Not At Risk", 1: "At Risk"}

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    st.divider()

    st.subheader("Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numerical Features", len(df.select_dtypes(include="number").columns))
    col4.metric("Categorical Features", len(df.select_dtypes(exclude="number").columns))
    st.divider()

    st.subheader("Average Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average GPA", round(df["GPA"].mean(), 2))
    col2.metric("Average Attendance", round(df["Attendance_Rate"].mean(), 2))
    col3.metric("Average Learning Hours", round(df["Online_Learning_Hours"].mean(), 2))
    col4.metric("Average Lab Performance", round(df["Lab_Performance"].mean(), 2))
    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.divider()

    # ----- Filters
    st.subheader("Filter Dataset")
    filtered_df = df.copy()
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["All"] + sorted(df["Gender"].dropna().unique().tolist()))
    with col2:
        socio = st.selectbox(
            "Socio Economic Status",
            ["All"] + sorted(df["Socio_Economic_Status"].dropna().unique().tolist()),
        )
    with col3:
        risk = st.selectbox(
            "Risk Status",
            ["All", 0, 1],
            format_func=lambda v: v if v == "All" else risk_labels[v],
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        gpa_min, gpa_max = float(df["GPA"].min()), float(df["GPA"].max())
        gpa_range = st.slider("GPA Range", gpa_min, gpa_max, (gpa_min, gpa_max))
    with col5:
        # BUG FIX: default used to be the MAX age, which hid almost every student
        age = st.slider("Minimum Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].min()))
    with col6:
        attendance = st.slider("Minimum Attendance", 0, 100, 0)

    if gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender]
    if socio != "All":
        filtered_df = filtered_df[filtered_df["Socio_Economic_Status"] == socio]
    if risk != "All":
        filtered_df = filtered_df[filtered_df["At_Risk"] == risk]
    filtered_df = filtered_df[filtered_df["GPA"].between(gpa_range[0], gpa_range[1])]
    filtered_df = filtered_df[filtered_df["Age"] >= age]
    if attendance > 0:  # rows with missing attendance are only dropped when a minimum is set
        filtered_df = filtered_df[filtered_df["Attendance_Rate"] >= attendance]

    st.success(f"Showing {len(filtered_df)} student(s) after applying filters.")
    st.dataframe(filtered_df, use_container_width=True)
    st.divider()

    # ----- Charts
    st.subheader("Student Risk Distribution")
    target_count = df["At_Risk"].map(risk_labels).value_counts().reset_index()
    target_count.columns = ["Risk Status", "Students"]
    fig = px.pie(target_count, values="Students", names="Risk Status", title="Risk Distribution", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("GPA Distribution")
    st.plotly_chart(px.histogram(df, x="GPA", title="GPA Distribution", nbins=20), use_container_width=True)

    st.subheader("Attendance Analysis")
    st.plotly_chart(
        px.histogram(df, x="Attendance_Rate", title="Attendance Rate Distribution"),
        use_container_width=True,
    )

    st.subheader("Gender Distribution")
    gender_count = df["Gender"].value_counts().reset_index()
    gender_count.columns = ["Gender", "Students"]
    st.plotly_chart(px.bar(gender_count, x="Gender", y="Students", title="Gender Distribution"), use_container_width=True)

    st.subheader("GPA vs Attendance")
    scatter_df = df.assign(Risk=df["At_Risk"].map(risk_labels))
    st.plotly_chart(
        px.scatter(
            scatter_df,
            x="Attendance_Rate",
            y="GPA",
            color="Risk",
            title="Attendance vs GPA",
            hover_data=["Age", "Gender"],
        ),
        use_container_width=True,
    )
    st.divider()

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include="number").corr()
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", aspect="auto"),
        use_container_width=True,
    )
    st.divider()

    st.subheader("Dataset Summary")
    st.dataframe(df.describe(), use_container_width=True)
    st.divider()

    st.subheader("Top 10 Students by GPA")
    st.dataframe(df.sort_values(by="GPA", ascending=False).head(10), use_container_width=True)
    st.divider()

    st.subheader("Risk Summary")
    risk_summary = df["At_Risk"].map(risk_labels).value_counts().reset_index()
    risk_summary.columns = ["Risk Status", "Count"]
    st.dataframe(risk_summary, use_container_width=True)
    st.divider()

    # Insights below are backed by the notebook EDA and the model's feature importances
    st.subheader("Key Insights")
    st.success(
        """
        - At-risk students have a lower average GPA (about 2.77 vs 3.07).
        - Attendance is the strongest signal: at-risk students average about 67% vs 78%.
        - Assignments submitted also differ clearly (about 7.7 vs 10.4 on average).
        - Attendance, GPA and assignments together account for roughly 96% of the model's decisions.
        - Lab performance, forum participation, online learning hours, gender and extracurricular
          activity show almost no difference between the two groups in this dataset.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <center>
        <h4>Student Performance Risk Prediction System</h4>
        <p>Developed using <b>Streamlit</b> &bull; <b>Python</b> &bull; <b>Scikit-learn</b> &bull; <b>Plotly</b></p>
        </center>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
elif menu == "Prediction":
    st.header("Student Risk Prediction")
    st.write("Enter the student's details below to predict whether the student is academically at risk.")
    st.caption("Input ranges match the data the model was trained on.")
    st.divider()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Student Information")
            age = st.number_input("Age", min_value=18, max_value=29, value=20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            socio = st.selectbox("Socio Economic Status", ["Low", "Medium", "High"])
            extracurricular = st.selectbox("Extracurricular Participation", ["Yes", "No"])
        with col2:
            st.subheader("Academic Information")
            gpa = st.number_input("GPA", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
            attendance = st.slider("Attendance Rate (%)", 50, 100, 75)
            # BUG FIX: training data has 0-19 assignments; the old slider allowed 0-100 (default 70)
            assignments = st.slider("Assignments Submitted", 0, 19, 10)
            learning = st.slider("Online Learning Hours", 0.0, 15.0, 7.5, step=0.5)
            forum = st.slider("Forum Participation", 0, 9, 4)
            lab = st.slider("Lab Performance", 0, 100, 50)
        submitted = st.form_submit_button("Predict Student Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            {
                "Age": [age],
                "Gender": [gender],
                "Socio_Economic_Status": [socio],
                "GPA": [gpa],
                "Attendance_Rate": [float(attendance)],
                "Assignments_Submitted": [assignments],
                "Online_Learning_Hours": [float(learning)],
                "Forum_Participation": [forum],
                "Lab_Performance": [lab],
                # training data stored this column as float (0.0 / 1.0)
                "Extracurricular_Participation": [1.0 if extracurricular == "Yes" else 0.0],
            }
        )

        processed = preprocessor.transform(input_df)
        prediction = model.predict(processed)
        probability = model.predict_proba(processed)
        risk_probability = probability[0][1] * 100
        safe_probability = probability[0][0] * 100

        st.divider()
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Student is At Academic Risk")
            st.progress(min(risk_probability / 100, 1.0))
            st.metric("Risk Probability", f"{risk_probability:.2f}%")
            st.warning(
                """
                **Recommendations**
                - Improve attendance.
                - Submit assignments on time.
                - Increase online learning hours.
                - Participate more in laboratory activities.
                - Engage in classroom discussions and forum participation.
                - Seek academic guidance from mentors.
                """
            )
        else:
            st.success("Student is Not At Academic Risk")
            st.progress(min(safe_probability / 100, 1.0))
            st.metric("Prediction Confidence", f"{safe_probability:.2f}%")
            st.info(
                """
                - Keep up the good work.
                - Maintain regular attendance.
                - Continue submitting assignments on time.
                - Keep your GPA consistent.
                - Participate actively in lab sessions.
                """
            )
        st.divider()
        st.caption(
            "This is a decision-support estimate, not a final judgement. The model catches about 54% of "
            "truly at-risk students, so a 'Not At Risk' result does not guarantee the student is fine."
        )

# ---------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------
elif menu == "Model Performance":
    st.header("Model Performance")
    st.write(
        "This page presents the evaluation results of the final **Gradient Boosting Classifier** "
        "after hyperparameter tuning, measured on a 20% hold-out test set (400 students)."
    )
    st.divider()

    st.subheader("Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{TEST_ACCURACY * 100:.2f}%")
    col2.metric("Precision (At Risk)", f"{TEST_PRECISION * 100:.0f}%")
    col3.metric("Recall (At Risk)", f"{TEST_RECALL * 100:.0f}%")
    col4.metric("F1 Score (At Risk)", f"{TEST_F1 * 100:.0f}%")
    col5.metric("ROC-AUC", f"{TEST_AUC:.4f}")
    st.caption(
        f"For reference, always predicting 'Not At Risk' would already score {BASELINE_ACCURACY * 100:.2f}% accuracy "
        "because the data is imbalanced (about 71% / 29%). 5-fold cross-validation gives "
        "79.6% ± 1.4%, so the real-world accuracy is expected to be around 80%."
    )
    st.divider()

    st.subheader("Confusion Matrix")
    cm_fig = px.imshow(
        CONFUSION,
        text_auto=True,
        x=["Not At Risk", "At Risk"],
        y=["Not At Risk", "At Risk"],
        labels=dict(x="Predicted", y="Actual", color="Students"),
        color_continuous_scale="Blues",
        aspect="auto",
    )
    st.plotly_chart(cm_fig, use_container_width=True)
    st.divider()

    st.subheader("Classification Report")
    report = pd.DataFrame(
        {
            "Class": ["Not At Risk", "At Risk"],
            "Precision": ["0.83", "0.72"],
            "Recall": ["0.92", "0.54"],
            "F1 Score": ["0.87", "0.62"],
            "Support": ["285", "115"],
        }
    )
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("Feature Importance")
    try:
        names = preprocessor.get_feature_names_out()
        base_cols = ["Gender", "Socio_Economic_Status", "Extracurricular_Participation"]
        rows = []
        for name, value in zip(names, model.feature_importances_):
            feat = name.split("__", 1)[1]
            for c in base_cols:
                if feat.startswith(c):
                    feat = c
            rows.append((feat, value))
        imp = pd.DataFrame(rows, columns=["Feature", "Importance"]).groupby("Feature", as_index=False).sum()
        imp = imp.sort_values("Importance")
        st.plotly_chart(
            px.bar(imp, x="Importance", y="Feature", orientation="h", title="Gradient Boosting Feature Importance"),
            use_container_width=True,
        )
    except Exception:
        st.info("Feature importance is unavailable for this model version.")
    st.divider()

    st.subheader("Model Interpretation")
    st.success(
        """
        - The tuned Gradient Boosting model reaches 80.75% test accuracy and 0.82 ROC-AUC.
        - Attendance Rate, GPA and Assignments Submitted drive almost all of its predictions.
        - Lab performance, forum participation and online learning hours have very little influence.
        - The model can help educators spot students who may need additional academic support early.
        """
    )
    st.divider()

    st.subheader("Strengths")
    st.markdown(
        """
        - Beats the majority-class baseline (80.75% vs 71.25%).
        - Training and test accuracy are close (80.5% / 80.75%), so it is not overfitting.
        - Robust ensemble algorithm that handles non-linear relationships.
        - Easy to deploy using Streamlit.
        """
    )
    st.divider()

    st.subheader("Limitations")
    st.markdown(
        """
        - Recall for 'At Risk' is 54%: about 46% of truly at-risk students are missed.
        - Performance depends on the quality of the training dataset (2,000 records).
        - Predictions are limited to the ten input features.
        - The model should support, not replace, academic decision-making.
        """
    )
    st.divider()

    st.subheader("Future Scope")
    st.markdown(
        """
        - Lower the decision threshold to catch more at-risk students (trade-off: more false alarms).
        - Integrate with a student information system.
        - Enable real-time prediction using live student data.
        - Add downloadable PDF reports and progress tracking.
        """
    )
    st.divider()

    st.subheader("Conclusion")
    st.info(
        "The **Gradient Boosting Classifier** was selected as the final model after comparing six algorithms. "
        "Its performance shows that machine learning can support early identification of academically at-risk "
        "students, while its limited recall means it should be used as a screening aid alongside human judgement."
    )

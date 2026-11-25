
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load dataset and models
@st.cache_data
def load_data():
    return pd.read_csv("dataset/cleaned_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

df = load_data()
model = load_model()

results = pd.read_csv("results/model_comparison.csv")
cv = pd.read_csv("results/cross_validation.csv")
tuning = pd.read_csv("results/tuning_results.csv")
test_data = pd.read_csv("results/test_data.csv")

# Dashboard title
st.title("🛡️ Network Intrusion Detection System")

st.markdown(
    "Machine Learning Based Network Traffic Analysis "
    "and Intrusion Detection"
)

st.divider()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Home",
        "Dataset Overview",
        "Data Visualization",
        "Model Performance",
        "Intrusion Prediction",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.info("MCA Data Science Project")

# Function to display graphs
def show_graph(filename, title):
    path = os.path.join("graphs", filename)
    st.subheader(title)

    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning("Graph not found.")

# 1. Home page
if page == "Home":

    st.header("Project Dashboard")

    total = len(df)
    normal = int((df["label"] == 0).sum())
    malicious = int((df["label"] == 1).sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total:,}")
    col2.metric("Normal Traffic", f"{normal:,}")
    col3.metric("Malicious Traffic", f"{malicious:,}")
    col4.metric(
        "Attack Percentage",
        f"{malicious / total * 100:.2f}%"
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        "This project uses machine learning to identify "
        "normal and malicious network traffic. "
        "Three classification models are trained and "
        "compared to select the best intrusion detection model."
    )

    col1, col2 = st.columns(2)

    with col1:
        show_graph(
            "class_distribution.png",
            "Network Traffic Distribution"
        )

    with col2:
        show_graph(
            "model_comparison.png",
            "Machine Learning Performance"
        )

    st.subheader("Models Used")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree")
    st.write("3. Random Forest")

# 2. Dataset overview
elif page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Input Features", 41)
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Attack Types")
    st.dataframe(
        df["attack_type"].value_counts()
        .reset_index(),
        use_container_width=True
    )

    st.download_button(
        "Download Cleaned Dataset",
        df.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# 3. Data visualization
elif page == "Data Visualization":

    st.header("Exploratory Data Analysis")

    graphs = [
        ("class_distribution.png",
         "Normal vs Malicious Traffic"),

        ("traffic_percentage.png",
         "Traffic Percentage"),

        ("protocol_distribution.png",
         "Protocol Distribution"),

        ("attack_distribution.png",
         "Top Network Attacks"),

        ("service_distribution.png",
         "Network Services"),

        ("protocol_traffic.png",
         "Protocol-wise Traffic"),

        ("flag_distribution.png",
         "Connection Flags"),

        ("correlation_heatmap.png",
         "Correlation Heatmap")
    ]

    for i in range(0, len(graphs), 2):

        col1, col2 = st.columns(2)

        with col1:
            show_graph(*graphs[i])

        if i + 1 < len(graphs):
            with col2:
                show_graph(*graphs[i + 1])

# 4. Model performance
elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    st.subheader("Model Comparison")

    st.dataframe(
        results.round(4),
        use_container_width=True
    )

    show_graph(
        "model_comparison.png",
        "Performance Comparison"
    )

    st.subheader("5-Fold Cross-Validation")

    st.dataframe(
        cv.round(4),
        use_container_width=True
    )

    show_graph(
        "cross_validation.png",
        "Cross-Validation Results"
    )

    st.subheader("Hyperparameter Tuning")

    st.dataframe(
        tuning,
        use_container_width=True
    )

    best = tuning.loc[tuning["Best CV F1"].idxmax()]

    st.success("Selected Best Model: " + best["Model"])

    st.subheader("Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        show_graph(
            "logistic_regression_confusion_matrix.png",
            "Logistic Regression"
        )

        show_graph(
            "random_forest_confusion_matrix.png",
            "Random Forest"
        )

    with col2:
        show_graph(
            "decision_tree_confusion_matrix.png",
            "Decision Tree"
        )

        show_graph(
            "best_model_confusion_matrix.png",
            "Best Model - Final Testing"
        )

    show_graph("roc_curve.png", "ROC Curve")

    st.subheader("Final Classification Report")

    with open("results/final_report.txt", "r") as file:
        st.code(file.read())

# 5. Intrusion prediction
elif page == "Intrusion Prediction":

    st.header("Network Intrusion Prediction")

    st.write(
        "Select an unseen test record and predict "
        "whether the network traffic is normal or malicious."
    )

    record = st.number_input(
        "Select Test Record Number",
        min_value=0,
        max_value=len(test_data) - 1,
        value=0,
        step=1
    )

    sample = test_data.iloc[[int(record)]].copy()

    actual = int(sample["label"].iloc[0])

    input_data = sample.drop(columns=["label"])

    st.subheader("Network Traffic Details")

    st.dataframe(
        input_data.T.rename(columns={
            input_data.index[0]: "Value"
        }),
        use_container_width=True
    )

    if st.button("Predict Network Traffic", type="primary"):

        prediction = int(model.predict(input_data)[0])

        probability = model.predict_proba(input_data)[0]

        st.divider()

        if prediction == 0:
            st.success("Prediction: NORMAL TRAFFIC")
        else:
            st.error("Prediction: MALICIOUS TRAFFIC")

        col1, col2 = st.columns(2)

        col1.metric(
            "Normal Probability",
            f"{probability[0] * 100:.2f}%"
        )

        col2.metric(
            "Malicious Probability",
            f"{probability[1] * 100:.2f}%"
        )

        st.write(
            "Actual Traffic:",
            "Normal" if actual == 0 else "Malicious"
        )

        if prediction == actual:
            st.success("Correct Prediction")
        else:
            st.warning("Incorrect Prediction")

    st.caption(
        "This is an academic demonstration using "
        "held-out dataset records, not live network monitoring."
    )

# 6. About project
elif page == "About Project":

    st.header("About the Project")

    st.write("""
    **Project Name:** Network Intrusion Detection System

    **Dataset:** NSL-KDD

    **Programming Language:** Python

    **Development Environment:** VS Code

    **Dashboard:** Streamlit

    **Machine Learning Models:**
    - Logistic Regression
    - Decision Tree
    - Random Forest

    **Techniques Used:**
    - Data Cleaning
    - Exploratory Data Analysis
    - Feature Encoding
    - Feature Scaling
    - Train-Test Splitting
    - 5-Fold Cross-Validation
    - Hyperparameter Tuning
    - Model Evaluation
    - Intrusion Prediction

    **Objective:**

    To identify malicious network traffic using
    machine learning and compare the performance
    of different classification algorithms.
    """)

    st.info(
        "Developed for academic and educational purposes."
    )

st.divider()
st.caption("Network Intrusion Detection | MCA Data Science Project")
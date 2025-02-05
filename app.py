import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 as components  # Import Streamlit components

# Page Configuration
st.set_page_config(page_title="Disease Prediction Dashboard", layout="wide", page_icon="🩺")

# Updated Canva Embed Code
components.html(
    """
    <div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
    padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
    border-radius: 8px; will-change: transform;">
        <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
          src="https://www.canva.com/design/DAGWnb97gtE/cyRhaMPCQMsU4BVn2uVU1Q/view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
        </iframe>
    </div>
    <a href="https://www.canva.com/design/DAGWnb97gtE/cyRhaMPCQMsU4BVn2uVU1Q/view?utm_content=DAGWnb97gtE&utm_campaign=designshare&utm_medium=embeds&utm_source=link" target="_blank" rel="noopener">Epidemics vs. Pandemics Quiz Presentation in White Photographic Style</a> by Tanmay Shinde
    """,
    height=600,  # Adjust overall height as needed
)

# Disease Prediction Section
st.title("Disease Prediction Dashboard")
st.write("""
This interactive dashboard uses machine learning to predict the likelihood of various diseases based on selected datasets.
""")

# File Upload Section
uploaded_file = st.file_uploader("Upload a disease dataset (CSV format)", type=["csv"])
if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Overview")
    st.write("**Dataset Shape:**", df.shape)
    st.write("**First 5 rows:**")
    st.dataframe(df.head())
    
    # Disease Prediction Model (e.g., RandomForest for demonstration)
    st.subheader("Model Training & Prediction")
    target_variable = st.selectbox("Select Target Variable for Prediction", df.columns)
    
    if target_variable:
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Model Evaluation
        st.write("**Model Evaluation**")
        st.text(classification_report(y_test, model.predict(X_test)))

        # Feature Importance
        st.write("**Feature Importance**")
        feature_importance = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance, y=X.columns, ax=ax)
        st.pyplot(fig)

# Footer
st.write("## About")
st.write("""
This app is designed for analyzing health datasets and predicting disease outcomes using machine learning models.
""")

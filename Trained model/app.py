import streamlit as st
import pandas as pd
from sklearn.externals import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CTG Fetal Health Classifier",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load model + scaler + features
# -----------------------------
MODEL_PATH = "xgb_ctg_tuned_model.joblib"
SCALER_PATH = "ctg_scaler.joblib"
FEATURES_PATH = "ctg_features.joblib"

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_list = joblib.load(FEATURES_PATH)
    return model, scaler, feature_list

try:
    model, scaler, feature_list = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.stop()

# -----------------------------
# Header Section
# -----------------------------
st.title("🩺 Fetal Health Prediction System")
st.markdown("""
This application uses machine learning to classify fetal health based on Cardiotocography (CTG) data.
""")

# Clinical labels info in a nice card
with st.expander("ℹ️ Clinical Label Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**1 = Normal**\nHealthy fetal state")
    with col2:
        st.warning("**2 = Suspect**\nRequires monitoring")
    with col3:
        st.error("**3 = Pathologic**\nRequires intervention")

st.divider()

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "📁 Upload CTG Data (CSV format)",
    type=["csv"],
    help="Upload a CSV file containing CTG measurements"
)

# -----------------------------
# Main Processing Logic
# -----------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Preview Section
        st.subheader("📊 Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Required Features", len(feature_list))
        
        st.dataframe(df.head(10), use_container_width=True)

        # Feature validation
        missing_cols = [c for c in feature_list if c not in df.columns]
        if missing_cols:
            st.error(f" Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Prepare input data
        X = df[feature_list]
        X_scaled = scaler.transform(X)

        # Make predictions
        with st.spinner("🔄 Analyzing fetal health data..."):
            preds = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)

        # Map predictions to clinical labels
        model_to_clinical = {0: 1, 1: 2, 2: 3}
        clinical_to_text = {1: "Normal", 2: "Suspect", 3: "Pathologic"}

        clinical_labels = [model_to_clinical[p] for p in preds]
        readable_labels = [clinical_to_text[c] for c in clinical_labels]

        # Create results dataframe
        result_df = df.copy()
        result_df["Predicted_Label"] = clinical_labels
        result_df["Predicted_Class"] = readable_labels
        result_df["Confidence_Normal"] = (probas[:, 0] * 100).round(2)
        result_df["Confidence_Suspect"] = (probas[:, 1] * 100).round(2)
        result_df["Confidence_Pathologic"] = (probas[:, 2] * 100).round(2)

        # Results Summary
        st.divider()
        st.subheader(" Prediction Summary")
        
        total_records = len(result_df)
        normal_count = sum(result_df["Predicted_Label"] == 1)
        suspect_count = sum(result_df["Predicted_Label"] == 2)
        pathologic_count = sum(result_df["Predicted_Label"] == 3)
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric("Total Cases", total_records)
        with summary_cols[1]:
            st.metric("Normal", normal_count)
        with summary_cols[2]:
            st.metric("Suspect", suspect_count)
        with summary_cols[3]:
            st.metric("Pathologic", pathologic_count)

        # Detailed Results Table
        st.subheader(" Detailed Predictions")
        st.dataframe(result_df, use_container_width=True, height=400)

        # Download button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download Predictions (CSV)",
            data=csv,
            file_name="ctg_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        # -----------------------------
        # SHAP Explainability (FIXED)
        # -----------------------------
        st.divider()
        st.subheader(" Model Explainability")
        
        with st.expander(" About SHAP Values", expanded=False):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** shows which features most influence predictions.
            - Red points indicate higher feature values
            - Blue points indicate lower feature values
            - Position on x-axis shows impact on prediction
            """)

        # Class selection for SHAP
        class_names = ["Normal", "Suspect", "Pathologic"]
        selected_class = st.selectbox(
            "Select class to explain:",
            options=class_names,
            index=2  # Default to Pathologic
        )
        
        class_idx = class_names.index(selected_class)

        with st.spinner(f"Generating SHAP explanations for {selected_class}..."):
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            
            # Handle multiclass SHAP values
            # shap_values is a list of arrays, one per class
            if isinstance(shap_values, list):
                shap_values_for_class = shap_values[class_idx]
            else:
                shap_values_for_class = shap_values[:, :, class_idx]
            
            # Create SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values_for_class,
                X,
                feature_names=feature_list,
                show=False,
                plot_size=None
            )
            plt.title(f"Feature Importance for {selected_class} Classification", 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Feature importance bar chart
            st.subheader(" Top Feature Importances")
            feature_importance = np.abs(shap_values_for_class).mean(axis=0)
            importance_df = pd.DataFrame({
                'Feature': feature_list,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(10)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.barh(importance_df['Feature'], importance_df['Importance'])
            ax2.set_xlabel('Mean |SHAP Value|', fontsize=12)
            ax2.set_title(f'Top 10 Features for {selected_class}', 
                         fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    # Welcome screen
    st.info(" Upload a CTG CSV file to begin analysis")
    
    st.markdown("""
    ### How to use this application:
    1. Prepare your CTG data in CSV format
    2. Upload the file using the file uploader above
    3. View predictions and confidence scores
    4. Explore SHAP explanations to understand model decisions
    5. Download results for further analysis
    
    ### Required Features:
    The CSV file must contain all features same as original CTG.csv file.

    """)

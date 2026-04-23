import streamlit as st
import joblib 
import numpy as np 
import pandas as pd

# Page configuration
st.set_page_config(page_title="ChurnGuard AI | Horizon", layout="wide", page_icon="🔮")

# ⚡ Pro-Tip: Cache the model so it doesn't reload on every single click
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("best_churn_model.pkl")
    return scaler, model

scaler, model = load_assets()

# Main Header
st.title("🔮 ChurnGuard AI")
st.markdown("### Professional-grade retention analytics for your customer base.")
st.write("Enter customer metrics below or upload a batch file to predict attrition risk using our fine-tuned Support Vector Machine.")
st.divider()

# Sidebar for global info and controls
with st.sidebar:
    st.header("About ChurnGuard")
    st.markdown("""
    **Built exclusively for the Horizon Competition.**
    
    This application bridges the gap between raw data and actionable business intelligence. 
    
    **The Engine:**
    Powered by a Support Vector Classifier (SVC) equipped with an optimized RBF kernel.
    
    ** The Math:**
    To combat the 'Imbalanced Data Trap', this model utilizes balanced class weights. We optimized the Precision-Recall tradeoff to ensure high-risk customers are caught before they leave.
    """)
    
    st.divider()
    # ⚙️ AI Sensitivity 
    # Hardcoded to 75% to perfectly balance Precision and Recall
    threshold = 0.75
# Main tabs
tab1, tab2 = st.tabs(["👤 Individual Analysis", "📁 Batch Processing"]) 

with tab1:
    st.subheader("Single Customer Prediction")
    st.info("Input a customer's current metrics to get an instant real-time risk assessment.")
    
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
    with c2:
        gender = st.selectbox("Customer Gender", ["Male", "Female"])
    with c3:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=130, value=10)

    monthlycharge = st.number_input("Monthly Charge ($)", min_value=30, max_value=150, value=70)

    # Indented correctly!
    if st.button("Analyze Risk", use_container_width=True):
        gender_selected = 1 if gender == "Female" else 0
        X = np.array([[age, gender_selected, tenure, monthlycharge]])
        X_scaled = scaler.transform(X)

        # Get the probability instead of a hard 0 or 1
        probs = model.predict_proba(X_scaled)[0]
        churn_prob = probs[1] # Probability of Churn

        st.divider()

        # Compare against the slider in the sidebar!
        if churn_prob >= threshold:
            st.error(f"###  Result: HIGH RISK")
            st.markdown(f"**Confidence Score:** {churn_prob*100:.1f}%")
            st.write("This customer exhibits strong attrition signals. Immediate retention action is recommended.")
        else:
            st.success(f"###  Result: LOW RISK")
            st.markdown(f"**Retention Probability:** {(1-churn_prob)*100:.1f}%")
            st.write("This customer's profile aligns with stable, long-term users.")

with tab2:
    st.subheader("📁 Bulk Customer Analysis")
    st.write("Upload a CSV with these exact columns: `Age`, `Gender`, `Tenure`, `MonthlyCharges`.")
    
    uploaded_file = st.file_uploader("Drop your CSV here", type="csv")
    
    if uploaded_file:
        try:
            # 1. Read the file
            df = pd.read_csv(uploaded_file)
            
            # 2. Process and rename to match training data
            processed_df = df.copy()
            processed_df['Gender'] = processed_df['Gender'].map({'Female': 1, 'Male': 0})
            
            # 3. Select columns in the EXACT order your scaler expects
            features = processed_df[['Age', 'Gender', 'Tenure', 'MonthlyCharges']]
            
            # 4. Scaling & Prediction Probabilities
            X_scaled = scaler.transform(features)
            batch_probs = model.predict_proba(X_scaled)[:, 1] # Get probabilities for all rows
            
            # 5. Apply the threshold from the slider
            df['Churn_Risk_Score'] = [f"{p*100:.1f}%" for p in batch_probs]
            df['Prediction'] = ["High Risk" if p >= threshold else "Low Risk" for p in batch_probs]
            
            st.divider()
            st.success(" Batch Analysis Complete!")

            # 6. Dashboard Metrics
            res_col1, res_col2 = st.columns(2)
            churn_count = int((batch_probs >= threshold).sum())
            
            res_col1.metric("Total Customers Analyzed", len(df))
            res_col2.metric("At-Risk Detected", churn_count, delta=f"{churn_count/len(df)*100:.1f}%", delta_color="inverse")

            # Display the full table
            st.dataframe(df, use_container_width=True)

            # 7. Download results
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Detailed Report (CSV)",
                data=csv_output,
                file_name="horizon_batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except KeyError as e:
            st.error(f"Column Name Error: I couldn't find the column named {e}. Check your CSV headers!")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
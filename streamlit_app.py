import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load model + scaler
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
metrics = joblib.load("metrics.pkl")

st.title("Car Insurance Claim Prediction")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Analysis", "Prediction"])

if page == "Introduction":
    st.write("This app predicts whether a policyholder will file a claim.")
    st.write("Models trained: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM.")
    st.write("Best model selected based on ROC-AUC.")

elif page == "Analysis":
    st.header("📊 Model Evaluation Dashboard")

    # Load metrics
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(metrics).T.drop(columns=["ConfusionMatrix"])  # metrics dict → DataFrame

    # Show metrics table
    st.subheader("Performance Summary")
    st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"))

    # Bar chart for F1 scores
    st.subheader("F1 Score Comparison")
    st.bar_chart(df["F1"])

    # Heatmap of all metrics
    st.subheader("Heatmap of Metrics")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Confusion matrix for best model
    st.subheader("Confusion Matrix (Best Model)")
    from sklearn.metrics import ConfusionMatrixDisplay

    best_model_name = max(metrics, key=lambda k: metrics[k]["ROC-AUC"])
    cm = metrics[best_model_name]["ConfusionMatrix"]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Claim", "Claim"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    st.info(f"✅ Best model selected: {best_model_name}")


elif page == "Prediction":
    st.write("### Predict Claim from Test Data")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    policy_id = st.selectbox("Select Policy ID", sample["policy_id"].values)
    row = test[test["policy_id"] == policy_id].drop(columns=["policy_id"])

    # Encode categorical features same way as training
    for col in row.select_dtypes(include="object").columns:
        row[col] = row[col].astype("category").cat.codes

    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    prob = model.predict_proba(row_scaled)[0][1]

    st.write(f"Prediction for {policy_id}: **{'Claim' if pred==1 else 'No Claim'}**")
    st.write(f"Probability of Claim: {prob:.2f}")
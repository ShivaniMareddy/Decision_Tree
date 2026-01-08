import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Bank Marketing Prediction", layout="centered")

# ---------------- LOAD CSS ----------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
# ---------------- TITLE ----------------
st.markdown("""
<div class="card">
    <h1>Bank Marketing Prediction</h1>
    <p>Using <b>Decision Tree (Rule Based Model)</b> to predict whether a customer should be contacted</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "bank_marketing_dataset.csv")
    return pd.read_csv(file_path)

df = load_data()

# ---------------- DATA PREVIEW ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SELECT FEATURES ----------------
features = [
    'age', 'job', 'balance', 'loan',
    'contact', 'duration', 'campaign',
    'previous', 'poutcome'
]

target = 'deposit'
df = df[features + [target]].copy()

# ---------------- ENCODE CATEGORICAL (STORE ENCODERS) ----------------
encoders = {}
categorical_cols = ['job', 'loan', 'contact', 'poutcome']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------- SPLIT ----------------
X = df.drop('deposit', axis=1)
y = df['deposit'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ----------------
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# ---------------- CONFUSION MATRIX ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Do Not Contact", "Contact"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PERFORMANCE METRICS ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("True Positive (TP)", TP)

c3, c4 = st.columns(2)
c3.metric("True Negative (TN)", TN)
c4.metric("False Positive (FP)", FP)

c5, c6 = st.columns(2)
c5.metric("False Negative (FN)", FN)
c6.metric("Total Predictions", len(y_test))

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- EXPLANATION BOX ----------------
st.markdown("""
<div class="card">
<h3>Confusion Matrix Meaning</h3>
<ul>
<li><b>TP:</b> Correctly identified customers to contact</li>
<li><b>TN:</b> Correctly identified customers not to contact</li>
<li><b>FP:</b> Non-interested predicted as interested</li>
<li><b>FN:</b> Interested predicted as non-interested</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Customer Subscription")

age = st.slider("Age", 18, 95, 30)
job = st.selectbox("Job", encoders['job'].classes_)
balance = st.slider("Account Balance", -2000, 100000, 1000)
loan = st.selectbox("Personal Loan", encoders['loan'].classes_)
contact = st.selectbox("Contact Type", encoders['contact'].classes_)
duration = st.slider("Last Call Duration (seconds)", 0, 5000, 200)
campaign = st.slider("Number of Contacts in this Campaign", 1, 50, 1)
previous = st.slider("Number of Previous Contacts", 0, 50, 0)
poutcome = st.selectbox("Previous Campaign Outcome", encoders['poutcome'].classes_)

# ---------------- ENCODE USER INPUT USING SAME ENCODERS ----------------
job_encoded = encoders['job'].transform([job])[0]
loan_encoded = encoders['loan'].transform([loan])[0]
contact_encoded = encoders['contact'].transform([contact])[0]
poutcome_encoded = encoders['poutcome'].transform([poutcome])[0]

input_data = np.array([
    age,
    job_encoded,
    balance,
    loan_encoded,
    contact_encoded,
    duration,
    campaign,
    previous,
    poutcome_encoded
]).reshape(1, -1)

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

result = "Contact this customer" if prediction == 1 else "Do NOT contact this customer"

st.markdown(f"""
<div class="prediction-box">
Prediction: <b>{result}</b><br>
Subscription Probability: <b>{probability:.2f}</b>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

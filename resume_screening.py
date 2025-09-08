import streamlit as st
import joblib
import docx2txt
import PyPDF2
import re

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function: extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function: clean text (same as training)
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9+#\. ]+', ' ', text)
    return text.lower()

# Streamlit UI
st.title("📄 Automated Resume Screening App")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = docx2txt.process(uploaded_file)

    st.subheader("Extracted Resume Text:")
    st.write(text[:500] + "...")  # show first 500 chars

    # Preprocess
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    # Prediction
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max() * 100

    st.subheader("Prediction Result:")
    st.write(f"**Predicted Job Role:** {pred}")
    st.write(f"**Fit Score:** {prob:.2f}%")

import streamlit as st
import fitz  # PyMuPDF for PDF processing
import re
import spacy
from transformers import pipeline

# Load NLP Model for Research Classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load SpaCy NER Model for PII Detection
nlp = spacy.load("en_core_web_sm")

# Expanded Research Categories
RESEARCH_CATEGORIES = [
    "Pharmaceutical Research",
    "Clinical Trials",
    "Drug Development",
    "Medical Science",
    "Biomedical Engineering",
    "Bioinformatics",
    "Artificial Intelligence in Healthcare",
    "Operations Research",
    "Supply Chain Management",
    "Logistics and Transportation",
    "Finance and Investment",
    "Risk Analysis",
    "Blockchain in Finance",
    "Software Engineering",
    "Machine Learning Research",
    "Cybersecurity",
    "Data Science",
    "Physics",
    "Chemical Engineering",
    "Material Science"
]

# PII Patterns (Regex-based detection)
PII_PATTERNS = {
    "Email": r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+",
    "Phone Number": r"\+?\d[\d -]{8,15}\d",
    "Address": r"\d{1,5}\s\w+\s\w+",
    "Name": None  # Detected using SpaCy's Named Entity Recognition
}

# Streamlit UI Configuration
st.set_page_config(page_title="Unstructured Data Classifier", layout="wide")
st.title("📑 Unstructured Data Classifier & PII Detector")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing file..."):
        # Extract text from PDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])

        st.subheader("📜 Extracted Text")
        st.text_area("PDF Content", text[:2000] + "..." if len(text) > 2000 else text, height=250)

        # 1️⃣ Research Classification
        st.subheader("🧠 Research Classification")
        with st.spinner("Classifying research topic..."):
            result = classifier(text[:2000], RESEARCH_CATEGORIES)  # Using first 2000 chars for better classification
            st.write(f"**Primary Research Topic:** {result['labels'][0]} (Score: {result['scores'][0]:.2f})")
            st.write(f"**Other Possible Topics:** {result['labels'][1]} (Score: {result['scores'][1]:.2f})")
        
        # 2️⃣ Personal Information Detection
        st.subheader("🔍 Personal Information Detection")
        pii_results = []

        # Regex-based PII detection
        for pii_type, pattern in PII_PATTERNS.items():
            if pattern:
                matches = re.findall(pattern, text)
                pii_results.extend([(pii_type, match) for match in set(matches)])

        # Name detection using SpaCy
        doc_nlp = nlp(text)
        for ent in doc_nlp.ents:
            if ent.label_ == "PERSON":
                pii_results.append(("Name", ent.text))

        if pii_results:
            for pii_type, pii_value in pii_results:
                st.write(f"🔹 **{pii_type}:** {pii_value}")
        else:
            st.success("No personal information detected.")

st.sidebar.markdown("---")
st.sidebar.info("🚀 Future Features: AWS S3 & SharePoint Integration, DICOM Support, Enhanced Image Classification.")

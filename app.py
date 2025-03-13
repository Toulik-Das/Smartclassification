import string
import nltk
import pdfplumber
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
from docx import Document
import pptx
from PIL import Image
import pytesseract
import io
import boto3
import pydicom
from transformers import pipeline
from spacy.lang.en import English
from pymongo import MongoClient

nltk.download('stopwords')

# BERT model for classification
bert_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Expanded document types
DOCUMENT_TYPES = [
    "Resume", "Research Paper", "Scientific Report", "Medical Report", "Financial Report",
    "Legal Document", "Business Proposal", "Technical Manual", "Educational Material"
]

# AWS S3 Configuration
s3_client = boto3.client("s3")

# MongoDB Atlas Connection using Streamlit Secrets
mongo_client = MongoClient(st.secrets["MONGO_URI"])
db = mongo_client["document_analysis"]
collection = db["classified_documents"]

# Text Extraction Functions
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return ' '.join([para.text for para in doc.paragraphs])

def extract_text_from_pptx(pptx_file):
    presentation = pptx.Presentation(pptx_file)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + " "
    return text.strip()

def extract_text_from_image(image_file):
    image = Image.open(io.BytesIO(image_file.read()))
    return pytesseract.image_to_string(image)

def extract_text_from_dicom(dicom_file):
    dataset = pydicom.dcmread(dicom_file)
    return str(dataset)

# Preprocessing
nlp = English()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# BERT Classifier

def classify_document(text):
    result = bert_classifier(text, DOCUMENT_TYPES)
    return result['labels'][0]

# Save Metadata to MongoDB
def save_to_mongo(filename, doc_type, tfidf_scores):
    collection.insert_one({
        "filename": filename,
        "document_type": doc_type,
        "tfidf_scores": tfidf_scores
    })

# Streamlit UI
def main():
    st.title("AI Document Classifier")

    uploaded_file = st.sidebar.file_uploader("Upload Document", type=['pdf', 'docx', 'pptx', 'png', 'jpg', 'jpeg', 'dcm'], key='file')

    if uploaded_file:
        file_type = uploaded_file.type

        if 'pdf' in file_type:
            text = extract_text_from_pdf(uploaded_file)
        elif 'officedocument.wordprocessingml' in file_type:
            text = extract_text_from_docx(uploaded_file)
        elif 'officedocument.presentationml' in file_type:
            text = extract_text_from_pptx(uploaded_file)
        elif 'image' in file_type:
            text = extract_text_from_image(uploaded_file)
        elif 'dicom' in file_type:
            text = extract_text_from_dicom(uploaded_file)
        else:
            st.error("Unsupported file type")
            return

        processed_text = preprocess_text(text)

        doc_type = classify_document(processed_text)

        vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names).T

        save_to_mongo(uploaded_file.name, doc_type, tfidf_scores.to_dict())

        st.subheader("Document Type Classification")
        st.write(f"**Document Type:** {doc_type}")

        st.subheader("Top TF-IDF Phrases")
        st.dataframe(tfidf_scores.sort_values(by=0, ascending=False).head(20))

if __name__ == "__main__":
    main()

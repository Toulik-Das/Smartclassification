import string
import nltk
import pdfplumber
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
from docx import Document
import pptx
from PIL import Image
import pytesseract
import io

nltk.download('stopwords')

# Expanded training data for better classification accuracy
DOCUMENT_TYPES = [
    "Resume", "Research Paper", "Scientific Report", "Medical Report", "Financial Report",
    "Legal Document", "Business Proposal", "Technical Manual", "Educational Material"
]

training_texts = [
    "Experience, education, skills, projects, certifications",  # Resume
    "Abstract, introduction, methodology, results, conclusion",  # Research Paper
    "Experiment, observation, analysis, hypothesis",  # Scientific Report
    "Patient details, diagnosis, treatment, medication",  # Medical Report
    "Balance sheet, profit and loss, financial statement",  # Financial Report
    "Contract, agreement, clauses, legal terms",  # Legal Document
    "Market analysis, financial projection, business strategy",  # Business Proposal
    "Hardware specifications, installation guide, troubleshooting",  # Technical Manual
    "Syllabus, learning objectives, course material"  # Educational Material
]

classifier = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=300, random_state=42))
classifier.fit(training_texts, DOCUMENT_TYPES)


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()


def extract_text_from_docx(docx_file):
    """Extract text from a DOCX file."""
    doc = Document(docx_file)
    return ' '.join([para.text for para in doc.paragraphs])


def extract_text_from_pptx(pptx_file):
    """Extract text from a PPTX file."""
    presentation = pptx.Presentation(pptx_file)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + " "
    return text.strip()


def extract_text_from_image(image_file):
    """Extract text from an image file."""
    image = Image.open(io.BytesIO(image_file.read()))
    return pytesseract.image_to_string(image)


def preprocess_text(text):
    """Preprocess text by removing punctuation, stopwords, and converting to lowercase."""
    text = text.replace('-', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def tfidf_vectorization(text_data):
    """Compute TF-IDF scores and return a sorted DataFrame."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    df_tfidf_sum = df_tfidf.sum(axis=0).reset_index()
    df_tfidf_sum.columns = ['Phrase', 'TF-IDF Score']
    df_tfidf_sorted = df_tfidf_sum.sort_values(by='TF-IDF Score', ascending=False).reset_index(drop=True)
    df_tfidf_sorted.index = df_tfidf_sorted.index + 1
    df_tfidf_sorted['Phrase'] = df_tfidf_sorted['Phrase'].str.title()
    return df_tfidf_sorted


def classify_document(text):
    """Classify the document type using the trained classifier."""
    return classifier.predict([text])[0]


def plot_bar_chart(df):
    """Generate a bar chart of TF-IDF scores."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    fig, ax = plt.subplots()
    ax.barh(df['Phrase'], df['TF-IDF Score'], color=colors)
    ax.set_xlabel('TF-IDF Score')
    ax.set_title('Top Words or Phrases by Score')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    ax.invert_yaxis()
    return fig


def main():
    """Streamlit app for document text analysis and visualization."""
    st.title("Document Analyzer with TF-IDF and Classification")

    uploaded_file = st.sidebar.file_uploader("Upload File", type=['pdf', 'docx', 'pptx', 'png', 'jpeg', 'jpg'], key='file')

    # Declare session variable for file reference and toggle
    if 'file_ref' not in ss:
        ss.file_ref = None
    if 'show_file' not in ss:
        ss.show_file = False

    if uploaded_file:
        ss.file_ref = uploaded_file

    if ss.file_ref:
        with st.spinner("Extracting text and processing..."):
            file_type = ss.file_ref.type

            if 'pdf' in file_type:
                text = extract_text_from_pdf(ss.file_ref)
            elif 'officedocument.wordprocessingml' in file_type:
                text = extract_text_from_docx(ss.file_ref)
            elif 'officedocument.presentationml' in file_type:
                text = extract_text_from_pptx(ss.file_ref)
            elif 'image' in file_type:
                text = extract_text_from_image(ss.file_ref)
            else:
                st.error("Unsupported file type")
                return

            processed_text = preprocess_text(text)
            text_data = [processed_text]
            df_tfidf_sorted = tfidf_vectorization(text_data)
            doc_type = classify_document(processed_text)
            plot = plot_bar_chart(df_tfidf_sorted.head(20))

        st.subheader("Document Type Classification")
        st.write(f"**Document Type:** {doc_type}")

        st.pyplot(plot)

        st.subheader("Top TF-IDF Phrases")
        st.dataframe(df_tfidf_sorted.head(20))

        if st.button("Toggle File View"):
            ss.show_file = not ss.show_file

        if ss.show_file and 'pdf' in file_type:
            binary_data = ss.file_ref.getvalue()
            pdf_viewer(input=binary_data, width=700)


if __name__ == "__main__":
    main()

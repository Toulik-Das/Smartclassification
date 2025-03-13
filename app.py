import string
import nltk
import pdfplumber
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

nltk.download('stopwords')


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = pdf.pages[0].extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()


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
    """Streamlit app for PDF text analysis and visualization."""
    st.title("TF-IDF Text Analyzer with PDF Viewer")
    st.write("Upload a PDF file to analyze and visualize the top-scoring phrases using TF-IDF.")

    # Declare session variable for PDF reference and toggle
    if 'pdf_ref' not in ss:
        ss.pdf_ref = None
    if 'show_pdf' not in ss:
        ss.show_pdf = False

    st.file_uploader("Upload PDF file", type=['pdf'], key='pdf')

    if ss.pdf:
        ss.pdf_ref = ss.pdf  # Backup uploaded PDF

    if ss.pdf_ref:
        with st.spinner("Extracting text and processing..."):
            text = extract_text_from_pdf(ss.pdf_ref)
            processed_text = preprocess_text(text)
            text_data = [processed_text]
            df_tfidf_sorted = tfidf_vectorization(text_data)
            plot = plot_bar_chart(df_tfidf_sorted.head(20))

        st.pyplot(plot)

        st.subheader("Top TF-IDF Phrases")
        st.dataframe(df_tfidf_sorted.head(20))

        if st.button("Toggle PDF View"):
            ss.show_pdf = not ss.show_pdf

        if ss.show_pdf:
            binary_data = ss.pdf_ref.getvalue()
            pdf_viewer(input=binary_data, width=700)


if __name__ == "__main__":
    main()

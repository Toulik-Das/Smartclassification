import string
import nltk
import pdfplumber
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from transformers import pipeline

nltk.download('stopwords')

image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-so400m-patch14-384")
labels = ['street sweeping', 'litter pickup', 'pothole repair', 'parking enforcement']

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def preprocess_text(text):
    text = text.replace('-', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    text = ' '.join(text.split())
    return text

def tfidf_vectorization(text_data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    df_tfidf_sum = df_tfidf.sum(axis=0).reset_index()
    df_tfidf_sum.columns = ['Phrase', 'TF-IDF Score']
    df_tfidf_sorted = df_tfidf_sum.sort_values(by='TF-IDF Score', ascending=False)
    df_tfidf_sorted = df_tfidf_sorted.reset_index(drop=True)
    df_tfidf_sorted.index = df_tfidf_sorted.index + 1
    df_tfidf_sorted['Phrase'] = df_tfidf_sorted['Phrase'].str.title()
    return df_tfidf_sorted

def plot_bar_chart(df):
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    fig, ax = plt.subplots()
    ax.barh(df['Phrase'], df['TF-IDF Score'], color=colors)
    ax.set_xlabel('TF-IDF Score')
    ax.set_title('Top Words or Phrases by Score')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    ax.invert_yaxis()
    return fig

def image_mod(image):
    outputs = image_classifier(image, candidate_labels=labels)
    result = {dic["label"]: dic["score"] for dic in outputs}
    return result

def main():
    st.title("TF-IDF Text & Image Analyzer")
    st.write("Upload a PDF file to analyze and visualize the top-scoring phrases in the document using TF-IDF.")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Extracting text and processing..."):
            text = extract_text_from_pdf(uploaded_file)
            processed_text = preprocess_text(text)
            text_data = [processed_text]
            df_tfidf_sorted = tfidf_vectorization(text_data)
            plot = plot_bar_chart(df_tfidf_sorted.head(20))
        
        st.pyplot(plot)
        
        st.subheader("Top TF-IDF Phrases")
        st.dataframe(df_tfidf_sorted.head(20))
    
    st.write("Upload an image to classify its category based on predefined labels.")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        with st.spinner("Classifying image..."):
            image_result = image_mod(uploaded_image)
        st.write("Classification Results:")
        st.json(image_result)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import re
import os
import nltk
import numpy as np
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="Professional IR System: TF-IDF vs BM25", layout="wide")

# =====================================
# POIN 1: TEXT PREPROCESSING (Setup)
# =====================================
@st.cache_resource
def init_nltk():
    nltk.download("stopwords", quiet=True)

init_nltk()
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    # Tokenization & Case Folding
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    # Stopword Removal & Stemming
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_documents(folder_path):
    data = []
    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' tidak ditemukan!")
        return pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            category = filename.split('_')[0]
            full_path = os.path.join(folder_path, filename)
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n\n')
                title = lines[0] if len(lines) > 0 else filename
                data.append({
                    'filename': filename,
                    'full_path': full_path,
                    'title': title,
                    'content': content,
                    'category': category
                })
    return pd.DataFrame(data)

df = load_documents("data_berita")

# =====================================
# POIN 2 & 5: INVERTED INDEX & FEATURE SELECTION
# =====================================
if not df.empty:
    with st.spinner('Building System Components...'):
        if 'clean_text' not in df.columns:
            df["clean_text"] = df["content"].apply(preprocess)
            
        # --- POIN 2: INVERTED INDEX ---
        # Membuat struktur data index untuk pencarian cepat
        inverted_index = defaultdict(set)
        for idx, doc in enumerate(df["clean_text"]):
            for term in doc.split():
                inverted_index[term].add(idx)
        
        # --- POIN 5: SELEKSI FITUR (TF-IDF Global) ---
        # Menyeleksi fitur kata menggunakan pembobotan TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
        feature_names = vectorizer.get_feature_names_out()

        # Setup BM25
        tokenized_corpus = [doc.split() for doc in df["clean_text"]]
        bm25_model = BM25Okapi(tokenized_corpus)

# =====================================
# POIN 6: SUMMARIZATION (Extractive)
# =====================================
def summarize(text, n_sentences=2):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= n_sentences: return text
    # Summarization menggunakan ranking TF-IDF per kalimat
    tfidf_sum = TfidfVectorizer(stop_words="english")
    try:
        matrix = tfidf_sum.fit_transform(sentences)
        scores = matrix.sum(axis=1).A1
        top_ids = scores.argsort()[::-1][:n_sentences]
        return " ".join([sentences[i] for i in sorted(top_ids)])
    except:
        return sentences[0]

# EVALUATION FUNCTION
def evaluate(results):
    if len(results) == 0: return 0, 0, "N/A"
    categories = [df.iloc[idx]["category"] for idx in results]
    majority_cat = Counter(categories).most_common(1)[0][0]
    precision = sum(1 for c in categories if c == majority_cat) / len(results)
    rr = 0
    for i, cat in enumerate(categories):
        if cat == majority_cat:
            rr = 1 / (i + 1)
            break
    return precision, rr, majority_cat

# =====================================
# POIN 3 & 4: IR MODEL & QUERY PROCESSING
# =====================================
st.title("📂 Professional Information Retrieval System")
query = st.text_input("Masukkan Query (Misal: 'apple technology' atau 'football match'):")

if query and not df.empty:
    # --- POIN 4: QUERY PROCESSING ---
    q_clean = preprocess(query)
    q_terms = q_clean.split()

    # --- POIN 3: IR MODEL (TF-IDF + Cosine Similarity) ---
    q_vec = vectorizer.transform([q_clean])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    
    # Ranking
    tfidf_ranked = tfidf_scores.argsort()[::-1][:5].tolist()
    tp, tmrr, tcat = evaluate(tfidf_ranked)

    # BM25 Ranking (Sebagai Pembanding)
    bm25_scores = bm25_model.get_scores(q_terms)
    bm25_ranked = bm25_scores.argsort()[::-1][:5].tolist()
    bp, bmrr, bcat = evaluate(bm25_ranked)

    # --- UI DISPLAY ---
    st.subheader("📊 Metrik Evaluasi")
    c1, c2 = st.columns(2)
    c1.metric("TF-IDF Precision@5", f"{int(tp*100)}%")
    c2.metric("BM25 Precision@5", f"{int(bp*100)}%")

    tab1, tab2 = st.tabs(["Hasil TF-IDF + Cosine", "Hasil BM25"])

    with tab1:
        for i, idx in enumerate(tfidf_ranked, 1):
            with st.expander(f"RANK {i} | {df.iloc[idx]['filename']} (Score: {round(tfidf_scores[idx], 4)})"):
                st.write(f"**Judul:** {df.iloc[idx]['title']}")
                # Menampilkan fitur kata terpenting untuk dokumen ini (Poin 5)
                row = tfidf_matrix.getrow(idx).toarray()[0]
                top_features = [feature_names[j] for j in row.argsort()[-3:][::-1]]
                st.write(f"**Fitur Kata Utama:** `{', '.join(top_features)}`")
                
                st.write(f"**Ringkasan Extractive:** {summarize(df.iloc[idx]['content'])}")
                if st.button(f"Lihat Full Text (TF-IDF {i})", key=f"t_{idx}"):
                    st.text(df.iloc[idx]['content'])

    with tab2:
        for i, idx in enumerate(bm25_ranked, 1):
            with st.expander(f"RANK {i} | {df.iloc[idx]['filename']} (Score: {round(bm25_scores[idx], 4)})"):
                st.write(f"**Ringkasan Extractive:** {summarize(df.iloc[idx]['content'])}")
                if st.button(f"Lihat Full Text (BM25 {i})", key=f"b_{idx}"):
                    st.text(df.iloc[idx]['content'])
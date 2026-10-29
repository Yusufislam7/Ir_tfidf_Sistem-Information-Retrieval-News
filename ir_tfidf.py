import pandas as pd
import re
import nltk
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# =====================================
# LOAD DATASET
# =====================================
df = pd.read_csv("bbc-news-data.csv", sep="\t")

# TITLE WEIGHTING (judul dikali 2)
df["text"] = (df["title"] + " ") * 2 + df["content"]

# =====================================
# PREPROCESSING
# =====================================
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess)

# =====================================
# INVERTED INDEX (DIPAKAI BENERAN)
# =====================================
inverted_index = defaultdict(set)
for idx, doc in enumerate(df["clean_text"]):
    for term in doc.split():
        inverted_index[term].add(idx)

# =====================================
# QUERY
# =====================================
query = input("\nMasukkan kata kunci pencarian: ")
query_clean = preprocess(query)
query_terms = query_clean.split()

# Ambil kandidat dokumen dari inverted index
candidate_docs = set()
for term in query_terms:
    candidate_docs |= inverted_index.get(term, set())

if not candidate_docs:
    candidate_docs = set(range(len(df)))

candidate_docs = list(candidate_docs)

# =====================================
# TF-IDF RETRIEVAL
# =====================================
vectorizer = TfidfVectorizer()  
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

query_vec = vectorizer.transform([query_clean])
tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

tfidf_ranked = sorted(candidate_docs, key=lambda i: tfidf_scores[i], reverse=True)[:5]

# =====================================
# BM25 RETRIEVAL
# =====================================
tokenized_docs = [doc.split() for doc in df["clean_text"]]
bm25 = BM25Okapi(tokenized_docs)
bm25_scores = bm25.get_scores(query_terms)

bm25_ranked = sorted(candidate_docs, key=lambda i: bm25_scores[i], reverse=True)[:5]

# =====================================
# EXTRACTIVE TEXT SUMMARIZATION
# =====================================
def summarize(text, n_sentences=2):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= n_sentences:
        return text

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_sent = tfidf.fit_transform(sentences)
    scores = tfidf_sent.sum(axis=1).A1

    top_ids = scores.argsort()[::-1][:n_sentences]
    summary = " ".join([sentences[i] for i in sorted(top_ids)])
    return summary

# =====================================
# EVALUATION (Precision@K & MRR)
# =====================================
def evaluate(results):
    categories = [df.iloc[idx]["category"] for idx in results]
    majority_category = Counter(categories).most_common(1)[0][0]

    relevant = 0
    rr = 0

    for i, idx in enumerate(results):
        if df.iloc[idx]["category"] == majority_category:
            relevant += 1
            if rr == 0: 
                rr = 1 / (i + 1)

    precision = relevant / len(results)
    return precision, rr, majority_category

tfidf_p, tfidf_mrr, tfidf_cat = evaluate(tfidf_ranked)
bm25_p, bm25_mrr, bm25_cat = evaluate(bm25_ranked)


# =====================================
# DISPLAY RESULTS
# =====================================
print("\n===== HASIL TF-IDF =====")
for i, idx in enumerate(tfidf_ranked, 1):
    print(f"\nRank {i}")
    print("Judul   :", df.iloc[idx]["title"])
    print("Kategori:", df.iloc[idx]["category"])
    print("Skor    :", tfidf_scores[idx])
    print("Ringkasan:")
    print(summarize(df.iloc[idx]["content"]))

print("\n===== HASIL BM25 =====")
for i, idx in enumerate(bm25_ranked, 1):
    print(f"\nRank {i}")
    print("Judul   :", df.iloc[idx]["title"])
    print("Kategori:", df.iloc[idx]["category"])
    print("Skor    :", bm25_scores[idx])
    print("Ringkasan:")
    print(summarize(df.iloc[idx]["content"]))

print("\n===== EVALUASI =====")
print(f"TF-IDF → Precision@5: {round(tfidf_p,2)} | MRR: {round(tfidf_mrr,2)} | Kategori dominan: {tfidf_cat}")
print(f"BM25   → Precision@5: {round(bm25_p,2)} | MRR: {round(bm25_mrr,2)} | Kategori dominan: {bm25_cat}")

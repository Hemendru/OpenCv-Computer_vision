import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.write("✅ App started")

nltk.download("stopwords")
nltk.download("wordnet")

# Load CSV (NO PICKLE)
df = pd.read_csv("movies_metadata.csv")

df = df[['title','overview','genres','vote_average','popularity','tagline']]
df['overview'] = df['overview'].fillna("")
df['tagline'] = df['tagline'].fillna("")
df['title'] = df['title'].astype(str)

def parse_genres(x):
    try:
        return " ".join([i['name'] for i in eval(x)])
    except:
        return ""

df['genres'] = df['genres'].apply(parse_genres)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[\W_]', ' ', text)
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    )

df['tags'] = (
    df['overview'] + " " + df['genres'] + " " + df['tagline']
).apply(preprocess)

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['tags'])

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommended(title, n=10):
    idx = indices[title]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    movie_indices = sim.argsort()[::-1][1:n+1]
    return df.iloc[movie_indices][['title','genres','vote_average','popularity']]

st.title("🎬 Movie Recommendation System")

movie = st.selectbox("Select a movie", sorted(df['title'].unique()))

if st.button("Recommend"):
    st.dataframe(recommended(movie))

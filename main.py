# Used libraries 
import pandas as pd
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cluster import KMeans
import re
import ast as ast

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


global df_artists, df_tracks

def main():
    global df_artists, df_tracks
    df_artists, df_tracks = read_data()
    df_artists = preprocessing_genre(df_artists)
    return df_artists, df_tracks



def read_data():
    df_artists = pd.read_csv('data/spotify_artists.csv')
    df_tracks = pd.read_csv('data/spotify_tracks.csv')
    df_artists = df_artists[df_artists['genres'] != '[]']
    return df_artists, df_tracks

def preprocessing_genre(df_artists):
    # Preparing genre data: Convertion from string to list
    df_artists = genre_toList(df_artists)

    # Preparing genre data: Cleaning
    df_artists = genre_cleaning(df_artists)

    # Preparing genre data: Tokenization
    df_artists = genre_tokenization(df_artists)

    # Stemming genres
    df_artists = genre_stemming(df_artists)       

    # Lemmatizing genres
    df_artists = genre_lemmatization(df_artists)  

    return df_artists

def genre_toList(df_artists):
    df_artists['genres'] = df_artists['genres'].apply(ast.literal_eval)
    return df_artists

def genre_cleaning(df_artists):
    df_artists['genres'] = df_artists['genres'].apply(lambda x: [genre.strip().lower() for genre in x])
    df_artists['cleaned_genres'] = df_artists['genres'].apply(lambda x: [re.sub(r'[^a-zA-Z\s]', '', genre) for genre in x])
    return df_artists

def genre_tokenization(df_artists):
    df_artists['tokenized_genre'] = df_artists['cleaned_genres'].apply(lambda x: [nltk.word_tokenize(genre) for genre in x])
    df_artists['flattened_tokens'] = df_artists['tokenized_genre'].apply(lambda x: [item for sublist in x for item in sublist])
    return df_artists

def genre_stemming(df_artists):
    stemmer = PorterStemmer()
    df_artists['stemmed_genres'] = df_artists['cleaned_genres'].apply(
        lambda x: [stemmer.stem(genre) for genre in x]
    )
    return df_artists

def genre_lemmatization(df_artists):
    lemmatizer = WordNetLemmatizer()
    df_artists['lemmatized_genres'] = df_artists['cleaned_genres'].apply(
        lambda x: [lemmatizer.lemmatize(genre) for genre in x]
    )
    return df_artists

def feature_engineering():
    # sollte dann befüllt werden
    a = 1+1
    return a


def genre_one_hot_encoding(df_artists):
    mlb = MultiLabelBinarizer()
    ohe_result = pd.DataFrame(mlb.fit_transform(df_artists['flattened_tokens']), columns=mlb.classes_)
    # Concatenate the original DataFrame with the new one-hot encoded DataFrame
    df_artists_ohe = pd.concat([df_artists, ohe_result], axis=1)
    return df_artists_ohe

def genre_bag_of_words(df_artists):
    # Create an instance of CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_artists['flattened_tokens'].apply(' '.join))
    # Create a DataFrame with the BoW
    bow_result = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    df_artists_bow = pd.concat([df_artists, bow_result], axis=1)
    return df_artists_bow

def genre_tfidf(df_artists):
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_artists['flattened_tokens'].apply(' '.join))
    # Create a DataFrame with the TF-IDF values
    tfidf_result = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    df_artists_tfidf = pd.concat([df_artists, tfidf_result], axis=1)
    return df_artists_tfidf

def preprocessing_lyrics():
    # sollte dann befüllt werden
    a = 1+1
    return a

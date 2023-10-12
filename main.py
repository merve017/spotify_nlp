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
    df_artists, df_tracks = read_data()
    preprocessing_genre()



def read_data():
    df_artists = pd.read_csv('data/artists.csv')
    df_tracks = pd.read_csv('data/tracks.csv')
    df_artists = df_artists[df_artists['genres'] != '[]']
    return df_artists, df_tracks

def preprocessing_genre():
    # Preparing genre data: Convertion from string to list
    genre_toList()

    # Preparing genre data: Cleaning
    genre_cleaning()

    # Preparing genre data: Tokenization
    genre_tokenization()

    # Stemming genres
    genre_stemming()       

    # Lemmatizing genres
    genre_lemmatization()  

def genre_toList():
    df_artists['genres'] = df_artists['genres'].apply(ast.literal_eval)

def genre_cleaning():
    df_artists['genres'] = df_artists['genres'].apply(lambda x: [genre.strip().lower() for genre in x])
    df_artists['cleaned_genres'] = df_artists['genres'].apply(lambda x: [re.sub(r'[^a-zA-Z\s]', '', genre) for genre in x])

def genre_tokenization():
    df_artists['tokenized_genre'] = df_artists['cleaned_genres'].apply(lambda x: [nltk.word_tokenize(genre) for genre in x])

def genre_stemming():
    stemmer = PorterStemmer()
    df_artists['stemmed_genres'] = df_artists['cleaned_genres'].apply(
        lambda x: [stemmer.stem(genre) for genre in x]
    )

def genre_lemmatization():
    lemmatizer = WordNetLemmatizer()
    df_artists['lemmatized_genres'] = df_artists['cleaned_genres'].apply(
        lambda x: [lemmatizer.lemmatize(genre) for genre in x]
    )

def preprocessing_lyrics():
    # sollte dann befüllt werden
    a = 1+1
    return a

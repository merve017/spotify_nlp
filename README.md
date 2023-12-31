# Spotify Trend Analysis

<img width="1423" alt="Bildschirmfoto 2023-06-21 um 18 40 55" src="https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png">


## Topic



## Research question


## Data
- spotify_albums.csv
- spotify_tracks.csv
- spotify_artists.csv
- tracks_spotify_lang.csv (contains lyrics data after language detection)

The data is not on github and can be found [here](https://we.tl/t-gr8kZE7Ar1) for download. For running the code, please create a folder data/ and put the csv files in it.

## EDA 
The EDA can be found in the following Notebook:

## How to run the code

1. Download the data. (see section Data)
2. Take a look at the requirements.txt and be sure that everything is installed/ready.
3. The Lyrics folder contains all notebooks that were used for lyrics-related work. The file      Get_Language_Translation determines the language of each song and has code that was used for trying to translate non-english lyrics. Furtermore, it creates the tracks_spotify_lang.csv, which is already in the data folder. The main script for lyrics related work is in Lyrics.ipnyb it contains the EDA, Feature Engineering, Topic Modelling and Sentiment Analysis. The file Preprocessing Lyric contains all the preprocessing steps and some extra feature engineering, with which we have experimented.

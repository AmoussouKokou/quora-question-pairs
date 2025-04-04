import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# Vérification et téléchargement des ressources nécessaires
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

class TextSimilarityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, use_tfidf=True):
        """
        Initialise l'extracteur de caractéristiques textuelles.
        :param use_tfidf: Si True, utilise TfidfVectorizer, sinon CountVectorizer.
        """
        self.use_tfidf = use_tfidf
        # Initialisation du vectoriseur en fonction de l'option choisie
        self.vectorizer = TfidfVectorizer(min_df=2) if use_tfidf else CountVectorizer()
    
    def fit(self, X, y=None):
        """
        Entraîne le vectoriseur sur l'ensemble des textes fournis.
        :param X: DataFrame avec les colonnes 'text1' et 'text2'.
        """
        # Concaténation des colonnes textuelles pour entraîner le vectoriseur
        texts = pd.concat([X.iloc[:, 0].dropna(), X.iloc[:, 1].dropna()], axis=0)
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, X):
        """
        Transforme les paires de textes en un ensemble de caractéristiques.
        :param X: DataFrame contenant les colonnes 'text1' et 'text2'.
        :return: DataFrame contenant les features calculées.
        """
        # Remplacement des valeurs manquantes par des chaînes vides
        X = X.fillna("")
        
        # Vectorisation des textes pour le calcul de similarité cosinus
        vec_texts = self.vectorizer.transform(X.iloc[:, 0]), self.vectorizer.transform(X.iloc[:, 1])
        cosine_sims = cosine_similarity(vec_texts[0], vec_texts[1]).diagonal()
        
        # Calcul des longueurs de texte
        len_text1, len_text2 = X.iloc[:, 0].str.len(), X.iloc[:, 1].str.len()
        len_diff = abs(len_text1 - len_text2)
        
        # Calcul du nombre de mots dans chaque texte
        num_words1, num_words2 = X.iloc[:, 0].str.split().str.len(), X.iloc[:, 1].str.split().str.len()
        num_words_diff = abs(num_words1 - num_words2)
        
        # Calcul de la densité lexicale (rapport entre mots et longueur totale du texte)
        lexical_density1 = num_words1 / (len_text1 + 1e-6)
        lexical_density2 = num_words2 / (len_text2 + 1e-6)
        lexical_density_diff = abs(lexical_density1 - lexical_density2)
        
        # Calcul du nombre de mots communs entre les deux textes
        common_words = X.apply(lambda row: len(set(row.iloc[0].split()) & set(row.iloc[1].split())), axis=1)
        common_words_ratio = common_words / (num_words1 + num_words2 - common_words + 1e-6)
        
        # Création du DataFrame final contenant toutes les caractéristiques
        features = pd.DataFrame({
            'len_text1': len_text1,
            'len_text2': len_text2,
            'len_diff': len_diff,
            'num_words1': num_words1,
            'num_words2': num_words2,
            'num_words_diff': num_words_diff,
            'lexical_density1': lexical_density1,
            'lexical_density2': lexical_density2,
            'lexical_density_diff': lexical_density_diff,
            'common_words': common_words,
            'common_words_ratio': common_words_ratio,
            'cosine_sim': cosine_sims
        })
        
        return features

if __name__ == "__main__":
    # Définition des textes à comparer
    data = pd.DataFrame({
        'text1': ["Hello, how are you?", "What is your name?", "The weather is nice today."],
        'text2': ["Hi, how are you?", "Tell me your name.", "It's a beautiful day."]
    })
    
    # Initialisation et exécution de l'extracteur de caractéristiques
    feature_extractor = TextSimilarityFeatures(use_tfidf=True)
    feature_extractor.fit(data)
    features = feature_extractor.transform(data)
    
    # Affichage des caractéristiques extraites
    print(features)

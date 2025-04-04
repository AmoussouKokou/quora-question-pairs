import numpy as np
import pandas as pd
import re
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# Vérification et téléchargement des ressources nécessaires pour les stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

class TextSimilarityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, use_tfidf=True, batch_size=5000, cache_model=False):
        """
        Initialise l'extracteur de caractéristiques textuelles.
        :param use_tfidf: Si True, utilise TfidfVectorizer, sinon CountVectorizer.
        :param batch_size: Taille des lots pour le traitement en batch.
        :param cache_model: Sauvegarde et recharge le modèle vectoriseur pour éviter de le recalculer.
        """
        self.use_tfidf = use_tfidf
        self.batch_size = batch_size
        self.cache_model = cache_model
        self.model_path = "vectorizer.pkl"
        self.vectorizer = None

    def fit(self, X, y=None):
        """
        Entraîne le vectoriseur sur l'ensemble des textes fournis.
        :param X: DataFrame avec les colonnes 'text1' et 'text2'.
        """
        texts = pd.concat([X.iloc[:, 0].dropna(), X.iloc[:, 1].dropna()], axis=0).unique()
        
        # Chargement du modèle vectoriseur s'il est déjà sauvegardé
        if self.cache_model:
            try:
                self.vectorizer = joblib.load(self.model_path)
                return self
            except FileNotFoundError:
                pass
        
        # Initialisation du vectoriseur selon l'option choisie (TF-IDF ou CountVectorizer)
        self.vectorizer = TfidfVectorizer(min_df=2) if self.use_tfidf else CountVectorizer()
        self.vectorizer.fit(texts)
        
        # Sauvegarde du modèle vectoriseur pour un futur usage
        if self.cache_model:
            joblib.dump(self.vectorizer, self.model_path)
        
        return self

    def transform(self, X):
        """
        Transforme les paires de textes en un ensemble de caractéristiques.
        :param X: DataFrame contenant les colonnes 'text1' et 'text2'.
        :return: DataFrame contenant les features calculées.
        """
        if self.vectorizer is None:
            raise ValueError("Le vectoriseur doit être entraîné avec fit() avant d'utiliser transform().")
        
        # Remplacement des valeurs manquantes par des chaînes vides
        X = X.fillna("")
        
        features_list = []
        for start in range(0, len(X), self.batch_size):
            batch = X.iloc[start:start + self.batch_size]
            
            # Vectorisation des textes
            vec_texts = self.vectorizer.transform(batch.iloc[:, 0]), self.vectorizer.transform(batch.iloc[:, 1])
            
            # Calcul de la similarité cosinus entre les paires de textes
            cosine_sims = cosine_similarity(vec_texts[0], vec_texts[1]).diagonal()
            
            # Longueur des textes
            len_text1, len_text2 = batch.iloc[:, 0].str.len(), batch.iloc[:, 1].str.len()
            len_diff = abs(len_text1 - len_text2)
            
            # Nombre de mots par texte
            num_words1 = batch.iloc[:, 0].str.split().str.len()
            num_words2 = batch.iloc[:, 1].str.split().str.len()
            num_words_diff = abs(num_words1 - num_words2)
            
            # Densité lexicale (nombre de mots / longueur du texte)
            lexical_density1 = num_words1 / (len_text1 + 1e-6)
            lexical_density2 = num_words2 / (len_text2 + 1e-6)
            lexical_density_diff = abs(lexical_density1 - lexical_density2)
            
            # Nombre de mots communs entre les deux textes
            common_words = batch.apply(lambda row: len(set(row.iloc[0].split()) & set(row.iloc[1].split())), axis=1)
            common_words_ratio = common_words / (num_words1 + num_words2 - common_words + 1e-6)
            
            # Création du DataFrame avec les caractéristiques
            features_list.append(pd.DataFrame({
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
            }))
        
        # Retourne le DataFrame final contenant toutes les caractéristiques
        return pd.concat(features_list, ignore_index=True)

if __name__ == "__main__":
    # Données d'exemple pour tester la classe
    data = pd.DataFrame({
        'text1': ["Hello, how are you?", "What is your name?", "The weather is nice today."],
        'text2': ["Hi, how are you?", "Tell me your name.", "It's a beautiful day."]
    })
    
    # Initialisation et exécution de l'extracteur de caractéristiques
    feature_extractor = TextSimilarityFeatures(use_tfidf=True, batch_size=2, cache_model=True)
    feature_extractor.fit(data)
    features = feature_extractor.transform(data)
    print(features)

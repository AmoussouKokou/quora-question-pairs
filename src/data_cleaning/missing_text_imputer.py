import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

class MissingTextImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.corpus = []  # Stocke les textes disponibles

    def fit(self, X, y=None):
        """
        Apprend le corpus de textes existants (hors valeurs manquantes) pour mesurer la similarité.
        """
        self.corpus = list(X.iloc[:, 0].dropna()) + list(X.iloc[:, 1].dropna())
        self.vectorizer.fit(self.corpus)
        return self
    
    def find_most_dissimilar(self, text, candidates):
        """
        Trouve le texte le plus différent d'un texte donné dans une liste de candidats.
        """
        if not candidates:
            return "MISSING_TEXT"
        
        # Vectorisation des textes
        vectors = self.vectorizer.transform([text] + candidates)
        similarities = cosine_similarity(vectors[0], vectors[1:]).flatten()
        
        # Sélection du texte avec la plus faible similarité
        return candidates[np.argmin(similarities)]

    def transform(self, X, y=None):
        """
        Remplace les valeurs manquantes selon la règle définie.
        """
        X = X.copy()
        candidates = self.corpus  # Tous les textes disponibles
        
        for i in X.index:
            text1, text2 = X.loc[i, X.columns[0]], X.loc[i, X.columns[1]]
            target = y[i] if y is not None else None
            
            if pd.isna(text1):
                if target == 1:
                    X.loc[i, X.columns[0]] = text2
                else:
                    X.loc[i, X.columns[0]] = self.find_most_dissimilar(text2, candidates)
            
            if pd.isna(text2):
                if target == 1:
                    X.loc[i, X.columns[1]] = text1
                else:
                    X.loc[i, X.columns[1]] = self.find_most_dissimilar(text1, candidates)
        
        return X
    
if __name__ == "__main__":
    # Création d'un DataFrame avec des valeurs manquantes
    data = pd.DataFrame({
        'text1': ["Bonjour", np.nan, "Comment ça va ?", "Merci beaucoup", np.nan],
        'text2': [np.nan, "Salut", np.nan, "Merci infiniment", "Au revoir"]
    })

    # Cible indiquant si les textes sont similaires (1) ou non (0)
    target = np.array([1, 0, 1, 1, 0])

    # Instanciation et utilisation du transformateur
    imputer = MissingTextImputer()
    imputer.fit(data)
    transformed_data = imputer.transform(data, target)

    print("Données après imputation :")
    print(transformed_data)

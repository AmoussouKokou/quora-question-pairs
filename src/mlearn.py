import pandas as pd
import numpy as np
import os
import download_data as dld
from sklearn.model_selection import train_test_split
# import re
# import string
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score
# from scipy.spatial.distance import cosine
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk


# # Télécharger les stopwords si nécessaire
# nltk.download("stopwords")
# nltk.download("punkt")

class Mlearn:
    def __init__(self, train_path="quora_data/train.csv", ):
        """Initialisation de la classe avec le chemin du fichier d'entraînement."""
        self.train_path = train_path
        # self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None # RandomForestClassifier(n_estimators=100, random_state=42)
        self.data = None

    def load_data(self):
        """Charge et nettoie les données."""
        print("🔹 Chargement des données...")

        if not os.path.exists(self.train_path):
            # Étape 1: Télécharger les données
            dld.download_kaggle_data()

            # Étape 2: Décompresser le fichier quora-question-pairs.zip dans quora_data
            dld.unzip_quora_zip()
            
            # Étape 3: Renommer le fichier test.csv en test2.csv
            dld.rename_test_file()

            # Étape 4: Décompresser les fichiers .zip (train.csv.zip, test.csv.zip, etc.) dans quora_data
            dld.unzip_files()

            # Étape 5: Supprimer les fichiers ZIP et les fichiers dézippés après extraction
            dld.delete_zip_files()
        
        train_data = pd.read_csv(self.train_path)
        train_data = train_data.dropna()  # Suppression des valeurs manquantes
        train_data = train_data.sample(frac=1).reset_index(drop=True)  # Mélange des données

        self.data = train_data
    
    def split_data(self):

        print("🔹 Découpage des données...")

        X = self.data[['question1', 'question2']]
        y = self.data['is_duplicate']

        # Découper en Train (70%), Validation (15%), Test (15%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val


    # def rndforest(self):
    #     """Entraînement du modèle RandomForestClassifier."""
    #     print("�� Entraînement du modèle RandomForestClassifier...")

    # def clean_text(self, text):
    #     """Nettoyage du texte (suppression des caractères spéciaux, mise en minuscules, etc.)."""
    #     text = text.lower()
    #     text = re.sub(f"[{string.punctuation}]", "", text)  # Suppression de la ponctuation
    #     words = word_tokenize(text)
    #     words = [word for word in words if word not in stopwords.words("english")]
    #     return " ".join(words)

    def extract_features(self, X):
        """Extraction de caractéristiques pour le machine learning."""
        print("🔹 Extraction des caractéristiques...")

        # Nettoyage des questions
        df["clean_q1"] = df["question1"].astype(str).apply(self.clean_text)
        df["clean_q2"] = df["question2"].astype(str).apply(self.clean_text)

        # Vectorisation TF-IDF
        questions = pd.concat([df["clean_q1"], df["clean_q2"]], axis=0).values
        self.vectorizer.fit(questions)

        q1_tfidf = self.vectorizer.transform(df["clean_q1"]).toarray()
        q2_tfidf = self.vectorizer.transform(df["clean_q2"]).toarray()

        # Calcul de la similarité cosinus entre TF-IDF des deux questions
        df["cosine_similarity"] = [1 - cosine(q1_tfidf[i], q2_tfidf[i]) for i in range(len(df))]

        # Différence de longueur
        df["length_diff"] = df["question1"].astype(str).apply(len) - df["question2"].astype(str).apply(len)

        return df[["cosine_similarity", "length_diff"]], df["is_duplicate"]

    # def train_model(self, X, y):
    #     """Entraîne le modèle de classification."""
    #     print("🔹 Entraînement du modèle...")
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     self.model.fit(X_train, y_train)
    #     y_pred = self.model.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)
    #     print(f"✅ Modèle entraîné ! Précision : {accuracy:.4f} | F1-score : {f1:.4f}")

    # def predict(self, question1, question2):
    #     """Prédit si deux questions sont des doublons."""
    #     clean_q1 = self.clean_text(question1)
    #     clean_q2 = self.clean_text(question2)

    #     # Transformation en TF-IDF
    #     q1_tfidf = self.vectorizer.transform([clean_q1]).toarray()[0]
    #     q2_tfidf = self.vectorizer.transform([clean_q2]).toarray()[0]

    #     cosine_similarity = 1 - cosine(q1_tfidf, q2_tfidf)
    #     length_diff = abs(len(question1) - len(question2))

    #     X_new = np.array([[cosine_similarity, length_diff]])
    #     prediction = self.model.predict(X_new)[0]
    #     return "Doublon" if prediction == 1 else "Non doublon"

    # def save_model(self, filename="quora_model.pkl"):
    #     """Sauvegarde le modèle entraîné."""
    #     joblib.dump((self.model, self.vectorizer), filename)
    #     print(f"💾 Modèle sauvegardé sous {filename}.")

    # def load_model(self, filename="quora_model.pkl"):
    #     """Charge un modèle sauvegardé."""
    #     self.model, self.vectorizer = joblib.load(filename)
    #     print(f"📂 Modèle chargé depuis {filename}.")

# 🔥 Utilisation de la classe
if __name__ == "__main__":
    detector = Mlearn()

    detector.load_data()

    detector.split_data()

    print(detector.X_train)
    
    # Étape 1: Chargement et préparation des données
    # data = detector.load_data()
    
    # # Étape 2: Extraction des caractéristiques
    # X, y = detector.extract_features(data)

    # # Étape 3: Entraînement du modèle
    # detector.train_model(X, y)

    # # Étape 4: Sauvegarde du modèle
    # detector.save_model()

    # # Exemple de prédiction
    # question1 = "What is the best way to learn machine learning?"
    # question2 = "How can I start learning ML?"
    # print(f"💡 Prédiction : {detector.predict(question1, question2)}")

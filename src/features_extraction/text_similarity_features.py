import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import jaccard
import nltk

# Vérification et téléchargement des ressources nécessaires
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

class TextSimilarityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, use_tfidf=True):
        """
        Initialise l'extracteur de caractéristiques textuelles.
        :param use_tfidf: Si True, utilise TfidfVectorizer, sinon CountVectorizer.
        """
        self.use_tfidf = use_tfidf
        self.vectorizer = TfidfVectorizer(min_df=2) if use_tfidf else CountVectorizer()
        
    def fit(self, X, y=None):
        """
        Entraîne le vectoriseur sur l'ensemble des textes fournis.
        :param X: DataFrame avec les colonnes 'text1' et 'text2'.
        """
        texts = list(X.iloc[:, 0].dropna()) + list(X.iloc[:, 1].dropna())
        self.vectorizer.fit(texts)
        return self
    
    def preprocess_text(self, text):
        """
        Nettoie le texte en supprimant la ponctuation et en tokenisant.
        :param text: Texte brut
        :return: Liste de tokens sans ponctuation
        """
        if not isinstance(text, str):
            return []
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())  # Supprime la ponctuation
        return word_tokenize(text)
    
    def transform(self, X):
        """
        Transforme les paires de textes en un ensemble de caractéristiques.
        :param X: DataFrame contenant les colonnes 'text1' et 'text2'.
        :return: DataFrame contenant les features calculées.
        """
        features = []
        
        for text1, text2 in zip(X.iloc[:, 0], X.iloc[:, 1]):
            
            # print(text1, text2)
            tokens1 = self.preprocess_text(text1)
            tokens2 = self.preprocess_text(text2)
            
            set1, set2 = set(tokens1), set(tokens2)
            set1_nostop, set2_nostop = set1 - stop_words, set2 - stop_words
            
            len_text1, len_text2 = len(text1), len(text2)
            len_diff = abs(len_text1 - len_text2)
            
            num_words1, num_words2 = len(tokens1), len(tokens2)
            num_words_diff = abs(num_words1 - num_words2)
            
            avg_word_len1 = np.mean([len(word) for word in tokens1]) if tokens1 else 0
            avg_word_len2 = np.mean([len(word) for word in tokens2]) if tokens2 else 0
            avg_word_len_diff = abs(avg_word_len1 - avg_word_len2)
            
            lexical_density1 = len(set1) / (num_words1 + 1e-6)
            lexical_density2 = len(set2) / (num_words2 + 1e-6)
            lexical_density_diff = abs(lexical_density1 - lexical_density2)
            
            common_words = len(set1 & set2)
            common_words_ratio = common_words / (len(set1 | set2) + 1e-6)
            
            all_words = list(set1 | set2)
            if all_words:
                vec1 = np.array([1 if word in set1 else 0 for word in all_words])
                vec2 = np.array([1 if word in set2 else 0 for word in all_words])
                jaccard_sim = 1 - jaccard(vec1, vec2)
            else:
                jaccard_sim = 0
            
            cosine_sim = cosine_similarity(
                self.vectorizer.transform([text1]), self.vectorizer.transform([text2])
            )[0][0]
            
            seq_ratio = SequenceMatcher(None, text1, text2).ratio()
            
            features.append([
                len_text1, len_text2, len_diff,
                num_words1, num_words2, num_words_diff,
                avg_word_len1, avg_word_len2, avg_word_len_diff,
                lexical_density1, lexical_density2, lexical_density_diff,
                common_words, common_words_ratio,
                jaccard_sim, cosine_sim,
                seq_ratio
            ])
            
        columns = [
            'len_text1', 'len_text2', 'len_diff',
            'num_words1', 'num_words2', 'num_words_diff',
            'avg_word_len1', 'avg_word_len2', 'avg_word_len_diff',
            'lexical_density1', 'lexical_density2', 'lexical_density_diff',
            'common_words', 'common_words_ratio',
            'jaccard_sim', 'cosine_sim',
            'seq_ratio'
        ]
        
        return pd.DataFrame(features, columns=columns)

if __name__=="__main__":
    import pandas as pd

    data = pd.DataFrame({
        'text1': ["Hello, how are you?", "What is your name?", "The weather is nice today."],
        'text2': ["Hi, how are you?", "Tell me your name.", "It's a beautiful day."]
    })

    feature_extractor = TextSimilarityFeatures(use_tfidf=True)
    feature_extractor.fit(data)
    features = feature_extractor.transform(data)
    
    print(features)

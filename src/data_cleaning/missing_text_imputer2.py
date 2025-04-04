import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MissingTextImputer(BaseEstimator, TransformerMixin):
    def __init__(self, dissimilar_word=" ", n_repeat=1):
        self.dissimilar_word = dissimilar_word
        self.n_repeat = n_repeat

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X):
        """
        Remplace les valeurs manquantes selon la règle définie.
        """
        X = X.copy()
        # if y is not None:
        #     y = np.asarray(y)

        fill_value = self.dissimilar_word * self.n_repeat

        mask_text1_missing = X.iloc[:, 0].isna()
        mask_text2_missing = X.iloc[:, 1].isna()
        
        # print(y is not None)
        # Si y == 1, on recopie l'autre texte
        if self.y is not None:
            X.loc[mask_text1_missing & (self.y == 1), X.columns[0]] = X.loc[mask_text1_missing & (self.y == 1), X.columns[1]]
            X.loc[mask_text2_missing & (self.y == 1), X.columns[1]] = X.loc[mask_text2_missing & (self.y == 1), X.columns[0]]

            # Si self.y == 0, on remplace par le mot dissemblable répété
            # print(X.loc[mask_text1_missing & (self.y == 0)])
            X.loc[mask_text1_missing & (self.y == 0), X.columns[0]] = fill_value
            X.loc[mask_text2_missing & (self.y == 0), X.columns[1]] = fill_value
            # print(X.loc[mask_text1_missing & (self.y == 0)])
        
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

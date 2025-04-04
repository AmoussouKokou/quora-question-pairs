import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
    roc_curve, auc, classification_report, precision_recall_curve, log_loss
)
from sklearn.utils.validation import check_is_fitted

class Evaluator(BaseEstimator, TransformerMixin):
    def __init__(self, modele=None):
        """
        Classe pour évaluer un modele scikit-learn.

        Parameters:
        -----------
        modele : sklearn.modele.modele
            modele contenant le modèle à évaluer.
        """
        self.modele = modele

    def fit(self, X=None, y=None):
        """Entraîne le modele.
        #todo: probleme avec l'etat de fit du modele
        """
        return self  # Respecte la convention sklearn

    def transform(self, X=None):
        """Applique la transformation du modele."""
        return X # self.modele.transform(X)

    
    def predict(self, y_true, y_pred, y_proba):
        """
        retourne les métriques sous forme de dictionnaire.

        Parameters:
        -----------
        X : array-like
            Données à prédire.
        y_true : array-like
            Vraies valeurs.

        Returns:
        --------
        dict
            Dictionnaire contenant accuracy, precision, recall, f1-score et classification report.
        """

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary"),
            "recall": recall_score(y_true, y_pred, average="binary"),
            "f1_score": f1_score(y_true, y_pred, average="binary"),
            "log_loss": log_loss(y_true, y_proba)
            # "classification_report": classification_report(y_true, y_pred, output_dict=True)
        }

        # Affichage des résultats
        print(f"🔹 Accuracy  : {metrics['accuracy']:.4f}")
        print(f"🔹 Precision : {metrics['precision']:.4f}")
        print(f"🔹 Recall    : {metrics['recall']:.4f}")
        print(f"🔹 log_loss    : {metrics['log_loss']:.4f}")
        print(f"🔹 F1-score  : {metrics['f1_score']:.4f}\n")

        # Matrice de confusion
        self.plot_confusion_matrix(y_true, y_pred)
        
        # roc
        self.plot_confusion_matrix(y_proba, y_true)

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """Affiche la matrice de confusion."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"])
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies classes")
        plt.title("Matrice de confusion")
        plt.show()
        
    def plot_roc_curve(self, y_proba, y_true):
        """Trace la courbe ROC et affiche l'AUC."""
        # y_proba = self.modele.predict_proba(X)[:, 1]  # Probabilité de la classe positive
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        sns.set(style="whitegrid", palette="muted")
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="dodgerblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
        plt.xlabel("Taux de faux positifs (FPR)", fontsize=12)
        plt.ylabel("Taux de vrais positifs (TPR)", fontsize=12)
        plt.title("Courbe ROC", fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()
        return roc_auc
    
    def plot_precision_recall_curve(self, X_test, y_test):
        """
        Calcule et affiche la courbe Precision-Recall avec Seaborn.

        Parameters:
        -----------
        X_test : array-like
            Données de test.
        y_test : array-like
            Labels réels de test.
        """
        # Prédictions des probabilités pour la classe positive
        y_scores = self.modele.predict_proba(X_test)[:, 1]  # Pour un problème binaire

        # Calcul de la courbe PR
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)

        # Création du DataFrame pour Seaborn
        import pandas as pd
        pr_df = pd.DataFrame({'Recall': recall, 'Precision': precision})

        # Affichage avec Seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=pr_df, x="Recall", y="Precision", label=f'PR Curve (AUC={pr_auc:.2f})')

        # Personnalisation du graphique
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        return pr_auc
        

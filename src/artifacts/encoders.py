import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer scikit-learn personnalisé qui applique la transformation logarithmique (log1p)
    aux colonnes spécifiées.
    """
    def __init__(self, columns=None):
        self.columns = columns  # Liste des colonnes à transformer

    def set_output(self, *, transform = None):
        # return super().set_output(transform=transform) 
        # soucis de version base estimator dans cette version low n'as oas cet attr
        pass
    
    def fit(self, X, y=None):
        """aucune opération d'entraînement nécessaire, retourne self."""
        return self

    def transform(self, X):
        """Applique np.log1p() aux colonnes spécifiées et retourne le DataFrame transformé."""
        X_transformed = X.copy()  # Éviter de modifier l'original
        if self.columns is None:
            raise ValueError("Veuillez spécifier les colonnes à transformer.")
        
        for col in self.columns:
            if col in X_transformed.columns:
                X_transformed[col] = np.log(X_transformed[col])  # log1p(x) = log(1 + x)
            else:
                raise ValueError(f"La colonne '{col}' n'existe pas dans les données.")

        return X_transformed

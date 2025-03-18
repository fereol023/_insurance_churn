import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from contents import *

def main():

    st.title("Généralisation aux données du hackathon")

    st.markdown(
        """
        ### Test de Kolmogorov-Smirnov

        Le test de KS pour deux échantillons est un test statistique non paramétrique utilisé pour comparer deux distributions empiriques. 
        Il permet de déterminer **si les deux échantillons proviennent de la même distribution** ou si leurs distributions diffèrent de manière significative.

        - Hypothèse nulle (H₀) : Les deux échantillons proviennent de la même distribution.

        - Pour chaque échantillon, on calcule une fonction de répartition empirique (ECDF), qui représente la proportion des données inférieures ou égales à une certaine valeur.
        
        - Statistique KS : est la distance maximale entre les deux fonctions de répartition empiriques. Elle est définie comme : D = max|F₁(x) - F₂(x)|
        Où F₁(x) et F₂(x) sont les fonctions de répartition empiriques des deux échantillons.

        - Une petite p-value (par exemple, < 0.05) indique que les distributions sont significativement différentes.

        - Applications : Comparer des distributions simulées et réelles (comme dans votre code).

        **Résultat entre targets**: La variable réelle labels correspond probablement à churn : p_value du test : 0.744 (KS stat = 0.0037)
        """
        )
    st.image(load('ressources/data/3_cleaned/', 'real_target_vs_simulated_target', 'png'))
    
    # dataviz
    st.markdown("### Comparer les features")

    st.markdown("Visualisation des distributions")
    st.markdown("Real db en brut")
    st.image(load('ressources/data/3_cleaned', 'realdb_raw', 'png'))
    st.markdown("Real db en log (variables numeriques)")
    st.image(load('ressources/data/3_cleaned', 'realdb_logged', 'png'))
    st.markdown("Db simulée au complet")

    st.markdown("Le test de KS a été refait entre les features (peu concluant car aucune correspondance sans transformation)")
    st.markdown("Matrice des p-valeurs")
    st.dataframe(load('ressources/data/3_cleaned/', 'ks_features_pvalues', 'pkl'))
    st.markdown("Aucune pvaleur n'est supérieure au seuil de 5% ; il n'y a pas assez de preuves pour affirmer que les features du hackathon et celles du dataset simulés proviennent du même échantillon.")
    st.markdown("Matrice des distances")
    st.dataframe(load('ressources/data/3_cleaned/', 'ks_features_distances', 'pkl'))
    st.markdown("Matrice des distances min")
    st.dataframe(load('ressources/data/3_cleaned/', 'ks_features_mindistances', 'pkl'))
    # test de correlation
    # prediction + lroc (new data)
    # metriques
    # conclusion
    st.markdown("""
        ### Conclusion et ouverture
        - *Avec les pistes explorées jusque là nous n'avons pas assez de preuves statistiques pour affirmer que les features proviennent de la même distribution et donc qu'elles encodent la même variable.*
        - *Normalement, les transformations sur de tels datasets ne modifient pas la distribution statistique des variables car on veut garder ces propriétes dans les modèles.*
        - *Soit, effectivement nous sommes en présence de nouvelles features ou alors nous faisons face à un data drift (changement de la distribution statistique pour une même variable en raison de facteurs exogènes).*
        - Pour pousser l'analyse il reste quelques pistes à explorer (pas fait faute de temps) à savoir :
                - des tests de correlation entre les features réelles et les features simulées
                - tenter de trouver des transformations pour retomber du des distributions similaires (exemple log mais en schiftant la distribution avant d'appliquer le log)
                - ou alors inverser les modalités des variables catégorielles etc..
                - on pourrait même faire un modèles se basant unuqment sur les vraiables socio demographiques en enlevant les variables comme la ville, le compté etc.. 
        """)

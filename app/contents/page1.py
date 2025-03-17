from contents import *
from unidecode import unidecode

var_selection_step1 = \
    """
    - *On calcule l'entropie, le taux de na et le taux de valeurs nulles sur les 23 colonnes du dataset.*
    - *On obtient que les variables `CURR_ANN_AMT` `INDIVIDUAL_ID` `SOCIAL_SECURITY_NUMBER` `ADDRESS_ID` ont une entropie elevee ce qui suppose une variabilite trop enorme dans les valeurs. Pour les 3 dernieres c'est normal car il s'agit de cles primaires. On les retire. En revanche, la premiere variable `CURR_ANN_AMT` represente la somme totale versee par l'assure au cours de l'annee precedente (ce n'est pas le montant de la police d'assurance). On la garde dans la mesure ou il s'agit d'une variable quantitaive qui pourrait s'averer utile pour la modelisation. A voir lors du feature engineering et de la data viz.*
    - *la variable `STREET_ADDRESS` contient les adresses des assures. Elle n'a pas 1 pour entropie car plusieurs assures peuvent resider a la meme adresse. Toutefois, en raison du fait qu'elle est une quasi cle primaire, on pourra la retirer egalement des variables d'interet car elle pourrait rapporter du bruit dans la modelisation.*
    - *La longitude et la latitude sont simules donc ils pourraient ne pas apporter d'information pertinente dans la modelisation mais seraient plutot utiles lors de la phase de dataviz.*
    - *`STATE` a une entropie nulle, elle ne varie pas et est tout le temps egale a TEXAS, a supprimer.*
    - *Enfin, la variable `ACCT_SUSPD_DATE` qui represente la date de suspension du contrat a quasiement 88% de valeurs manquantes. On On la supprimera apres l'etape de dataviz - elle nous servira a creer notre variable target (88% de personnes n'ayant pas resilie) - en effet lorsque la date d'annulation n'est pas renseignee, le contrat est encore en cours.*
    - *On passe donc de 23 a 17 variables d'interet a la fin de cette etape.*
    """

var_selection_step2 = \
    """
    #### VALEURS MANQUANTES

    - *Par la suite on va regarder en detail les autres variables qui ont des valeurs manquantes :*
    - `AGE` : 160k assures dont l'age est Nan - on peut essayer une imputation/(recalcul car on a la `DATE OF BIRTH` pas de null) dans la mesure ou ca ne represente que 5% du dataset. Mais attention parce qu'il y a des ages abberants (exple 112 ans pour un conducteur de voiture). On fera l'imputation seulement apres la dataviz.
    - `INCOME`: 160k (meme nombre que age) - a voir si ce sont les memes - strategie d'imputation aussi sinon
    - `HAS_CHILDREN`: 160k aussi sachant que les modalites yes/no sont bien renseignees - ne pas faire d'imputation
    - `LENGTH_OF_RESIDENCE` et `HOME_OWNER` et `COLLEGE_DEGREE` et `GOOD_CREDIT`: 160k 
    - `MARITAL STATUS`: une proportion de None differente (599k) - encoder par 'non renseigne'
    - `HOME MARKET VALUE`: une proportion differente (357k) - tenter imputation si dataviz ok (car on regrade dans un meme perimetre geographique) - ou predire avec les caracteriqtiques geographiques

    #### Conclusion partielle :
    *D'apres les colonnes qui manquent, on peut conclure que les 160k sont des assures dont on n'a pas les vraiables socio demographiques 
    On aurait pu tenter de predire la variable `HOME_MARKET_VALUE` a partir des autres mais il manque aussi les cacteristiques de la maison (nas).
    On peut tenter de faire une imputation pour garder la data coute que coute mais on ne fera pas. Plutot faire un drop car cela represente juste 5% du dataset.
    Une imputation ne rajouterait pas de la qualite au dataset.*
    """

var_selection_step3 = \
    """
    #### Sélection de variables
    Une fois la target cree, on supprime les dates car on a des variables derivees exple : DAYS_TENURE, AGE, CHURN et btw on est en 2022.
    """

    # Rappel
    # --------------------------------------------

    # *Il reste deux colonnes `LONGITUDE` et `LATITUDE` qui ont 20% de valeurs manquantes environs.*
    # Puisque la data est simulee, on peut les imputer par la moyenne de la longitude et de la latitude respectivement.
    # On peut aussi les imputer par la mediane ou la valeur la plus frequente.
    # On va faire une imputation pour pouvoir representer et eventuellement voir si elles ont une utilite dans la modelisation.*

def main():

    st.title('Traitement du dataset brut simulé')
    
    ROOT = 'ressources/data/1_raw/'
    st.markdown(
        """
        ### Aperçu des variables dans la database
        Le dataset contient 4 tables dont les varaibles sont les suivantes :
        """
    )
    st.write(load(ROOT, 'raw_variables_and_tables', 'pkl'))

    st.markdown("Pour reconstituer un dataset à partir des tables, nous avons fait des jointures successives. Voici le code : ")
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    st.image(load(ROOT, 'join_code_pic', 'png'))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("Le dataset reconstitué est le suivant : ")
    df = load(ROOT, 'head_df_full', 'parquet')
    st.dataframe(df)

    st.markdown("Voic une description du dataset à l'état brut/ou des types") # à refaire
    st.dataframe(load('ressources/data/3_cleaned/', 'data_description', 'pkl'))

    conclusion = """
    ### Conclusion partielle (reconstruction du dataset)

    - On est parti de 4 tables pour arriver à une seule table de 2 280 321 lignes et 22 colonnes. 
    - La table finale utilise 420MB de mémoire vive donc x5 pour la traiter avec pandas donc cette partie a été faite avec dask.
    """

    st.markdown(conclusion)

    st.title("Nettoyage du df - feature engineering")
    ROOT = 'ressources/data/2_intermediary/'

    st.title("Nettoyage") 

    _1, _2, _3 = st.columns(3)
    _2.image(load('ressources/data/1_raw/', 'insights_df_raw', 'png'))
    st.markdown(var_selection_step1)
    st.markdown(var_selection_step2)

    # ------ dataviz 
    st.markdown("Toutes les variables sont clean sauf l'age")
    col1, col2 = st.columns(2)
    col1.image(load(ROOT, 'boxplot_age_raw', 'png'))
    col2.image(load(ROOT, 'boxplot_age_cleaned', 'png'))
    st.markdown("""
                **Conclusion partielle :** 
                - Les variables sont clean sauf l'age (valeur manquantes et valeurs aberrantes), les variables socio démographiques sont manquantes aussi pour 160k personnes.
                - On a supprimé les personnes dont les âges sont trop grands (plus grands que 80 ans) car on suppose que ces ages sont des valeurs abérrantes au regard de la problématique.
                - De plus on a pas gardé les lignes dont les personnes ont leur valeurs socio démographiques (AGE, INCOME, HAS_CHILDREN, MARITAL_STATUS) manquantes (~5% du dataset)                
                """)
    st.markdown('### Feature engineering')
    # home market value
    st.markdown("## Home market value encoding")
    _1, _2, _3 = st.columns(3)
    _2.image(load('ressources/data/3_cleaned/', 'home_market_value', 'png'))
    st.dataframe(load(ROOT, 'encoding_home_market_value', 'pkl'))
    # target creation 
    st.markdown("""
                ## Target creation : Résilition (1:oui / 0:non) - churn
                LOGIQUE : les personnes dont la date est dispo/renseignee sont celles qui ont resilie. 
                Si la date de suspension est manquante (NA) c'est que la personne est toujours en portefeuille.
                On peut donc creer une variable target churn = 1 si la date est renseignee et churn = 0 sinon."""
                )
    _1, _2, _3 = st.columns(3)
    _2.image(load(ROOT, 'target_resiliation_creation', 'png'))

    st.markdown("#### ANOVA Resultats")
    _1, _2, _3 = st.columns(3)
    _ = ['CURR_ANN_AMT', 'DAYS_TENURE', 'AGE_IN_YEARS', 'INCOME']
    _2.table(load(ROOT, 'anova_results', 'pkl').dropna().loc[_])
    st.markdown(
        """
        #### Interpretation :

        Hypothèse H0 : les moyennes entre les groupes (catégories) sont égales

        - si p-value < 0.05, on rejette l'hypothèse nulle (H0) càd les moyennes des deux groupes sont différentes
        - si p-value > 0.05, on ne rejette pas l'hypothèse nulle (H0) càd les moyennes des deux groupes sont similaires

        *Puisque la pvaleur est partout nulle, on conclut que les variables numériques ci-dessu ne suffisent pas à elles seules à discriminer les personnes qui résilient de celles qui ne le font pas.*
        """
    )

    # test de khi deux

    st.markdown(var_selection_step3)

    # ---- apercu pipeline de numerisation 
    st.markdown("### Apercu du pipeline de numérisation")
    st.write(load('ressources/encoders_fitted/', 'numerisation_pipeline', 'pkl'))



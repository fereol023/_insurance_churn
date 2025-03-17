import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

import sys
sys.path.append('.')
sys.path.append('..')

from src.artifacts.model import ModelPipelineWithMlflow
from utils.fonctions import load_pickle

def load_db():
    return pd.read_parquet(r'ressources\data\0_archive\Insurance_Churn_ParticipantsData\train\data-balanced-rus.parquet').drop(['LONGITUDE', 'LATITUDE'], axis=1)


# ---------------------------
TARGET = 'churn'

#------- dataset
churn_data = load_db()
print("Dataset loaded ok !")

#------- encoders

# V1 --- standard scale les variables numeriques vues + garder les autres variables y.c longitude, latitude 
_encoders_pipeline = ColumnTransformer(
    transformers=[
        ('standard_scaler', StandardScaler(), ['INCOME', 'CURR_ANN_AMT', 'AGE_IN_YEARS']), # car dans les variables on a des ordres de grandeurs différents
    ], remainder='passthrough'
)

# V2 --- standard scaler les variables numeriques (days tenures y.c) + drop longitude, latitude
_encoders_pipeline_v2 = ColumnTransformer(
    transformers=[
        ('standard_scaler', StandardScaler(), ['INCOME', 'CURR_ANN_AMT', 'AGE_IN_YEARS', 'DAYS_TENURE']), # car dans les variables on a des ordres de grandeurs différents
    ], remainder='passthrough'
)
_encoders_pipeline.set_output(transform='pandas')
_encoders_pipeline_v2.set_output(transform='pandas')

#------- modele
def make_model_pipeline_with_cv_in_mlflow(sklearn_model, params, encoders_pipeline):
    """Fonction qui retourne une instance de pipeline avec mlflow juste à run."""
    model_cv = RandomizedSearchCV( # plus efficcace et moins couteurx que gridseacrhcv
        sklearn_model,
        param_distributions=params,
        scoring=None, # if none estimators method is used (pas good si on compare les estimateurs après, mettre le/les même scorer)
        cv=3, n_jobs=-1, verbose=2, n_iter=5 # vs 10 iter par defaut (iter = nbre de combinaisons choisies aux hasard dans la matrice des params)
    )
    
    model_pipeline = Pipeline([
        ('scaling', encoders_pipeline),
        ('model', model_cv)
    ])

    return ModelPipelineWithMlflow(
        model=model_pipeline,
        data=churn_data,
        target_name=TARGET,
        task="classification",
        experiment_name="modeles_churn"
    )

#-------- modeles candidats
candidates_models = [
        ('LogisticRegression_v2', LogisticRegression(), {'C': [0.001, 0.1, 1, 10, 20, 30], 'max_iter': [1_000], 'solver': ['lbfgs', 'saga']}),
        # ('KNeighbors_v2', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 17, 50, 100]}),
        ('RandomForest_v2', RandomForestClassifier(), {'max_depth': [100, 1_000, 10_000, None], 'n_estimators': [int(x) for x in np.linspace(1, 100, 5)], 'max_features': ['log2', 'sqrt'] }), # nb features à considérer racine_carree(16) ou 16 
        ('GradientBoosting_v2', GradientBoostingClassifier(), {'n_estimators': [int(x) for x in np.linspace(1, 100, 5)]}),
        ('AdaBoost_v2', AdaBoostClassifier(), {'n_estimators': [int(x) for x in np.linspace(1, 100, 5)]})
    ]

candidates_models_v2 = [
    *candidates_models,
    ('LightGBM_v2', LGBMClassifier(), {'num_leaves': [31, 50, 100], 'n_estimators': [int(x) for x in np.linspace(1, 100, 5)], 'learning_rate': [0.01, 0.1, 0.5]})
]

if __name__=='__main__':

    import time

    for model_name, model, param_grid in candidates_models_v2:
        # pour chaque modèle, run une experience
        pipeline = make_model_pipeline_with_cv_in_mlflow(
            model, 
            param_grid, 
            encoders_pipeline=_encoders_pipeline
        )
        print("="*100)
        print(f"Running pipeline for : {model_name}")
        pipeline.run(my_run_name=f"rus-scaling-and-{model_name}-wo_coord", save_shap=False, verbose=True)
        time.sleep(10)
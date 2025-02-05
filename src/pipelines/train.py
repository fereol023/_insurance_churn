import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV

import sys
sys.path.append('.')

from src.artifacts.model import ModelPipelineWithMlflow
from src.artifacts.encoders import LogTransformer
from utils.fonctions import load_pickle

def load_dataset(fpath, columns_to_read=[], indexes=None):
    data = pd.read_parquet(fpath, columns=columns_to_read)
    if indexes is not None:
        data = data.loc[indexes]
    if columns_to_read:
        data_ = data[list(columns_to_read)]
        del data
    return data_

# ---------------------------
metadata = load_pickle('ressources/data/3_cleaned/2025_01_15/variables_finales.pkl')
DATA_PATH = 'ressources/data/3_cleaned/2025_01_15/enedis_ban_ademe_extract_PARIS_2022.parquet'
TARGET_N_FEATURES = set(metadata.get('target') + metadata.get('features_nc'))
TARGET = metadata.get('target')
SAMPLE_IDX = metadata.get('idx_rows')
# print(metadata)

#------- dataset
enedis_data = load_dataset(DATA_PATH, TARGET_N_FEATURES, SAMPLE_IDX)
enedis_data.drop(['methode_application_dpe_ademe'], axis=1, inplace=True)
print("Dataset loaded ok !")

#------- encoders
num_not_as_quant = TARGET + ['version_dpe_ademe']
quant_features_names = [c for c in enedis_data.select_dtypes(include=['float', 'int']).columns if c not in num_not_as_quant]

encoders_pipeline = ColumnTransformer(
    transformers=[
        # ('standard_scaler', StandardScaler(), quant_features_names),
        # ajouter peut etre un truc pour log()
        
        ('log_transformer', LogTransformer(columns=quant_features_names), quant_features_names),
        ('ordinal_encoder', OrdinalEncoder(), ['etiquette_dpe_ademe'])
        # ajouter labelencoder 
    ], remainder='passthrough'
)

encoders_pipeline.set_output(transform='pandas')

#------- modele
param_rf = {'min_samples_split': [int(x) for x in np.linspace(10_000, 100_000, 2)]} # ajuster les params en fonction du fait qu'on a une valeur moyenne pour * logements pour y
model_pipeline_cv = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_rf, cv=2, n_jobs=-1, verbose=2, n_iter=1) # 10 iters par defaut, GrdiSearchCV + efficace aussi possible mais plus lent

model = Pipeline([
    ('encoding', encoders_pipeline),
    ('model', model_pipeline_cv)
])
    
pipeline = ModelPipelineWithMlflow(
    model=model,
    data=enedis_data,
    target_name=TARGET,
    task="regression",
    experiment_name="1_modele_non_contraint_conso_enedis"
)

if __name__=='__main__':
    pipeline.run(verbose=True)
    # mlflow ui in cli to launch
    # ou mlflow server --host --port
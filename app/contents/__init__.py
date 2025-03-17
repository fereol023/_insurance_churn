import json, os, pickle
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def fonction_communes_pages():
    pass

@st.cache
def load(root, filename, format):
    if format == 'png':
        _ = Image.open(f"{root}/{filename}.{format}")

    if format == 'parquet':
        _ = pd.read_parquet(f"{root}/{filename}.{format}")
    with open(os.path.join(root, f'{filename}.{format}'), 'rb') as f:
        if format == 'pkl':
            _ = pickle.load(f)
        elif format == 'txt':
            _ = f.read()

    return _


import mlflow
from mlflow.tracking import MlflowClient

def reload_model_by_run_name(search_run_name=None, exp_name="modeles_churn", tracking_uri="http://127.0.0.1:5000"):
    """Contrainte : le serveur doit etre up"""

    # Set the tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get the experiment by name
    experiment = client.get_experiment_by_name(exp_name)
    experiment_id = experiment.experiment_id

    # Search for the run by name
    runs = client.search_runs(experiment_id, filter_string=f"tags.mlflow.runName = '{search_run_name}'")
    run = runs[0]

    # Extract the run ID
    run_id = run.info.run_id

    # Load the model from the MLflow run
    model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/model_fitted/"
    try:
        lm = mlflow.sklearn.load_model(model_uri)
        print(f"Model {search_run_name} reloaded from MLflow experiment")
        st.markdown(f"Model {search_run_name} reloaded from MLflow experiment")
        return lm
    except Exception as e:
        print(f"Exception while loading model : {e}")
        st.markdown(f"Exception while loading model : {e}")

def reload_pipeline_by_name(name):
    
    PATH = f'ressources/models_fitted/{name}/pipeline_steps/'
    with open(os.path.join(PATH, "model/model.pkl"), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(PATH, 'scaling/scaling.pkl'), 'rb') as f:
        encoder = pickle.load(f)
        
    _pipeline = Pipeline(
        [('scaling', encoder), ('model', model)]
    )
    return encoder, model, _pipeline

def reload_with_run_id(id):
    logged_model = f'runs:/{id}/model_fitted'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model
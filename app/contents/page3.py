import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def main():

    # connexion au serveur MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  
    client = MlflowClient()

    st.title("Visualisation des expériences MLflow")

    # Récupérer toutes les expériences et afficher
    experiments = client.search_experiments()
    experiment_names = [exp.name for exp in experiments]
    selected_experiment_name = st.selectbox("Sélectionnez une expérience", experiment_names)

    # afficher les runs de l'expérience sélectionnée
    selected_experiment = next(exp for exp in experiments if exp.name == selected_experiment_name)
    runs = client.search_runs(selected_experiment.experiment_id, order_by=["start_time DESC"])

    # résultats sous forme de tableau
    if runs:
        runs_data = []
        for run in runs:
            runs_data.append({
                "Run ID": run.info.run_id,
                "Date": run.info.start_time,
                **run.data.params,  # paramètres
                **run.data.metrics,  # métriques
            })
        
        # tableau Streamlit
        runs_df = pd.DataFrame(runs_data)
        st.dataframe(runs_df)

        # Run pour explorer plus en détail
        selected_run_id = st.selectbox("Sélectionnez un Run ID", runs_df["Run ID"])
        if selected_run_id:
            selected_run = client.get_run(selected_run_id)
            
            st.subheader("Paramètres")
            st.write(selected_run.data.params)

            st.subheader("Métriques")
            st.write(selected_run.data.metrics)
            
            st.subheader("Artefacts")
            artifacts = client.list_artifacts(selected_run_id)
            for artifact in artifacts:
                st.write(artifact.path)
    else:
        st.write("Aucun run trouvé pour cette expérience.")

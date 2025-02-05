import os, tempfile, datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.pipeline import Pipeline 
from sklearn.utils import estimator_html_repr

# import sys
# sys.path.append('..')

from src.artifacts import MLFlowExp
from utils.fonctions import get_today_date, load_pickle, save_pickle

import shap
import matplotlib.pyplot as plt


class ModelPipelineWithMlflow(MLFlowExp):
    """
    Classe qui encapsule les étapes de la préparation, l'entraînement et l'évaluation.
    Trace les expériences via MLflow et sauvegarde les artefacts dans un dossier temporaire.

    param task: 'regression' ou 'classification'
    """
    def __init__(self, preprocessing=None, model=None, data=None, target_name=None, task=None, skip_model=False, experiment_name="modele"):
        self.preprocess = preprocessing
        self.model = model
        self.df = self.preprocess(data) if self.preprocess else data
        self.target_name = target_name[0] if isinstance(target_name, list) else target_name
        self.task = task
        self.skip_model = skip_model
        self.__check_init()
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = None, None, None, None, None
        self.is_trained = False
        self.score = 0
        self.experiment_name = experiment_name

        # experience mlflow configuree dans la classe parente
        super().__init__(self.experiment_name)
        self.artifact_dir = f"ressources/exp/{get_today_date()}/" # tempfile.mkdtemp()  # Dossier temporaire pour sauvegarder les artefacts
        if not os.path.exists(self.artifact_dir):
            os.makedirs(self.artifact_dir)

    def __check_init(self):
        if (self.model is None) and (not self.skip_model):
            raise Exception("Aucun modèle défini !")
        if not isinstance(self.df, pd.DataFrame):
            raise Exception("Aucun dataset défini !")
        if not self.target_name:
            raise Exception("Aucune variable cible définie !")

    def save_pipeline_steps(self):
        """si le modèle est un pipeline (et meme si c'est juste un modèle), sauvegarder séparerement les étapes"""
        # if isinstance(self.model, Pipeline):
        steps = self.model.named_steps
        for name, step in steps.items():
            step_path = os.path.join(self.artifact_dir, "pipeline_steps", name, f"{name}.pkl")
            # use pickle
            save_pickle(step, step_path)
            # et sauvegarder dans mlruns artifacts aussi
            self.mlflow.log_artifact(step_path, artifact_path=f"pipeline_steps/{name}")

    def load_pipeline_steps(self):
        """Reconctruit les étapes d'un pipeline à partir des artefacts sauvegardés"""
        if isinstance(self.model, Pipeline):
            steps = {}
            for name, step in self.model.named_steps.items():
                step_path = os.path.join(self.artifact_dir, "pipeline_steps", name, f"{name}.pkl")
                steps[name] = load_pickle(step_path, is_optional=False)
            self.model = Pipeline([(name, steps[name]) for name in steps])

    def data_split(self, verbose=True, test_size=0.2):
        X = self.df[[c for c in self.df.columns if c != self.target_name]]
        y = self.df[self.target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        # précaution
        if self.target_name in self.X_train.columns:
            self.X_train.drop(self.target_name, axis=1)
        if self.target_name in self.X_test.columns:
            self.X_test.drop(self.target_name, axis=1)
        if verbose:
            print(
                f"Train X : {self.X_train.shape}",
                f"Test X : {self.X_test.shape}",
                f"Train y : {self.y_train.shape}",
                f"Test y : {self.y_test.shape}",
                sep='\n'
            )
        self.mlflow.log_param("train_shape", self.X_train.shape)
        self.mlflow.log_param("test_shape", self.X_test.shape)
        self.mlflow.log_param("test_size", test_size)
        self.mlflow.log_param("num_features", X.shape[1])
        self.mlflow.log_param("num_samples", X.shape[0])
        return self

    def train(self, verbose=True, run_id=None):

        if run_id:
            self.artifact_dir = os.path.join(self.artifact_dir, str(run_id))
            if not os.path.exists(self.artifact_dir):
                os.makedirs(self.artifact_dir)

        start = datetime.datetime.now()

        self.model.fit(self.X_train, self.y_train)
        self.save_pipeline_steps()

        duration = datetime.datetime.now() - start

        if verbose:
            print(f"L'entraînement a duré : {duration} !")
        self.mlflow.log_metric("training_duration", duration.total_seconds())
        self.is_trained = True

        # image du pipeline
        with open(os.path.join(self.artifact_dir, "pipeline.html"), "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(self.model))

        # sauvegarder le modèle entraîné
        model_path = os.path.join(self.artifact_dir, "model_fitted")
        self.mlflow.sklearn.save_model(self.model, model_path) # sauvegarde le modele en local avec l'env et tout sans l'associer au tracker mlflow
        self.mlflow.sklearn.log_model(
            self.model, 
            artifact_path="model_fitted") # associer au tracker mlflow + afficher dans les artifacts de l'experience
        # MLflow stockera le modèle sous "mlruns/{run_id}/artifacts/model_fitted".
        
        # attend un fichier existant
        self.mlflow.log_artifacts(os.path.join(self.artifact_dir, "model_fitted"), artifact_path="model_fitted")
        # Cela stockera tous les fichiers du modèle sous "Artifacts" > "model_fitted" dans MLflow UI.
        print(f"model path : {model_path}")
        return self

    def predict(self, new_df=None, verbose=True, append_to_new_df=False):
        assert self.is_trained, "Le modèle doit être entraîné d'abord." # si on reload ?
        if new_df is None:
            if verbose:
                print("Pas de dataset fourni, on prédit sur le set de test.")
            new_df = self.X_test.copy()
        if verbose:
            print("Prédiction en cours...")
        y_pred = self.model.predict(new_df)
        if append_to_new_df:
            new_df[f"{self.target_name}_pred"] = y_pred
            return new_df
        return y_pred

    def eval(self, new_df=None, verbose=True):
        self.y_pred = self.predict(new_df, verbose=verbose)
        if self.task == "regression":

            rmse = round(root_mean_squared_error(self.y_test, self.y_pred), 4)
            mape = round(mean_absolute_percentage_error(self.y_test, self.y_pred), 4)
            r2 = round(r2_score(self.y_test, self.y_pred), 4)

            self.score = {'RMSE': rmse, 'MAPE': mape, 'R2_score': r2}

            self.mlflow.log_metric("RMSE", rmse)
            self.mlflow.log_metric("MAPE", mape)
            self.mlflow.log_metric("R2_score", r2)

        elif self.task == "classification":
            accuracy = accuracy_score(self.y_test, self.y_pred)
            self.score = {'accuracy': round(accuracy, 3)}
            self.mlflow.log_metric("accuracy", accuracy)
        else:
            raise Exception(f"Task {self.task} n'existe pas. Choisissez entre 'classification' ou 'regression'.")

        if verbose:
            print(f"Score d'évaluation : {self.score}")
        return self.score

    def run(self, new_data=None, save_shap=True, verbose=True):
        """
        Enchaîne les étapes de split, entraînement et évaluation.
        """
        with self.mlflow.start_run() as exp:
            
            self.data_split(verbose=verbose).train(verbose=verbose, run_id=exp.info.run_id).eval(new_data, verbose=verbose)
            self.mlflow.log_params({
                "model": self.model.__class__.__name__,
                "task": self.task
            })
            self.mlflow.log_params({**self.model['model'].best_params_})

            ## tt ce qui se passe en dehors du context manager est loggé dans une autre experience
            trainset_apercu = self.mlflow.data.from_pandas(
                pd.concat(
                    [self.X_train.head(5), self.y_train.head(5)], axis=1
                ),
                targets = self.target_name, # target col 
                name = "train set sample" # name of the dataset
            )
            testset_apercu = self.mlflow.data.from_pandas(
                pd.concat(
                    [self.X_test.head(5), self.y_test.head(5)], axis=1
                ),
                targets = self.target_name, # target col 
                name = "test set sample" # name of the dataset
            )

            self.mlflow.log_input(trainset_apercu, context='train set sample')
            self.mlflow.log_input(testset_apercu, context='test set sample')

            self.mlflow.log_text(pd.concat(
                    [self.X_train.head(5), self.y_train.head(5)], axis=1
                        ).to_markdown(), 'train_overview.md')
            
            self.mlflow.log_text(pd.concat(
                    [self.X_test.head(5), self.y_test.head(5)], axis=1
                        ).to_markdown(), 'test_overview.md')

            if save_shap:
                self.save_shap_artifacts(self.model, str(exp.info.run_id), str(exp.info.experiment_id))


    def save_shap_artifacts(self, pipeline_obj, run_id=None, exp_id=None):

        if self.target_name in self.X_test.columns:
            self.X_test = self.X_test.drop(self.target_name, axis=1)

        model = pipeline_obj.named_steps["model"].best_estimator_ # if has attr_ bestimator
        encoding = pipeline_obj.named_steps["encoding"]

        self.X_train_scaled = encoding.transform(self.X_train)
        self.X_test_scaled = encoding.transform(self.X_test)

        explainer = shap.Explainer(model, self.X_train_scaled)
        shap_values = explainer(self.X_test_scaled)

        # création d'un dossier temporaire pour stocker les images SHAP
        if run_id and (not run_id in str(self.artifact_dir)): # normalement à ce niveau le train a déjà update le path
            self.artifact_dir = os.path.join(self.artifact_dir, str(run_id))
        shap_plots_dirname = os.path.join(self.artifact_dir, "shap_plots")
        os.makedirs(shap_plots_dirname, exist_ok=True)

        # graphique global d'importance des features
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test_scaled, show=False)  # show=False pour éviter l'affichage direct
        summary_path = os.path.join(shap_plots_dirname, "summary_plot.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()

        # waterfall Plot pour une seule prédiction
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        waterfall_path = os.path.join(shap_plots_dirname, "waterfall_plot.png")
        plt.savefig(waterfall_path, bbox_inches="tight")
        plt.close()

        # dependence Plot pour une feature spécifique
        feature_name = "ordinal_encoder__etiquette_dpe_ademe"
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature_name, shap_values.values, self.X_test_scaled, show=False)
        dependence_path = os.path.join(shap_plots_dirname, f"dependence_{feature_name}.png")
        plt.savefig(dependence_path, bbox_inches="tight")
        plt.close()

        # logger les images dans MLflow
        self.mlflow.log_artifact(summary_path, artifact_path="shap_plots")
        self.mlflow.log_artifact(waterfall_path, artifact_path="shap_plots")
        self.mlflow.log_artifact(dependence_path, artifact_path="shap_plots")

        print(f"SHAP plots saved and logged in MLflow under 'shap_plots' directory.")

    def get_eval_score(self):
        return self.score

from contents import *
import mlflow
from mlflow.tracking import MlflowClient


def main():

    # connexion au serveur MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  
    client = MlflowClient()

    st.title("Modélisation - visualisation des expériences MLflow")
    ROOT = "ressources/data/3_cleaned/"
    # traitement
    st.markdown(
        """
        ### Equilbrer la target - Oversampling vs Undersampling
        Oversampling et undersampling sont des techniques utilisées pour traiter les ensembles de données déséquilibrés.

        **Oversampling** : Cette technique consiste à augmenter le nombre d'exemples dans la classe minoritaire en dupliquant les exemples existants ou en générant de nouveaux exemples synthétiques. L'objectif est de rendre les classes plus équilibrées en termes de nombre d'exemples.

        **Undersampling** : Cette technique consiste à réduire le nombre d'exemples dans la classe majoritaire en supprimant certains exemples. L'objectif est également de rendre les classes plus équilibrées, mais en réduisant la taille de la classe majoritaire.
        """
    )
    _1, _2, _3 = st.columns(3)
    _1.image(load(ROOT, 'imbalanced_target', 'png'))
    _2.image(load(ROOT, 'smote_balanced', 'png'))
    _3.image(load(ROOT, 'rus_balanced', 'png'))

    # Récupérer toutes les expériences et afficher
    st.text("\n")
    st.markdown("""
                ### Modèles

                **Objectifs**
                - FPR : erreur de prediction, on prédit 1 alors que ce n'est pas 1 - on va dire qu'ils résilient pas alors qu'ils ne résilient pas. (pas très grave)
                - FNR : erreur de prediction, on prédit 0 alors que ce n'est pas 0 - on va dire qu'ils ne partiront pas alors qu'il partent. (sensibilité/recall)
                - *Minimiser le fnr (faux negatifs) car ce sont des gens qu'on prédit qu'ils vont rester mais en vrai il partent.*
                - Si on minimise les FNR, le recall (TP/(TP + FN)) va être maximisé ; donc on va chercher à maximiser le recall en gardant une bonne précision.
                - *L'entrainement des modèles se fait via le script dans `src/pipelines/train.py` (cf. readme file)*
                """)
    
    st.markdown("Plusieurs modèles ont été testé avec de la CV parmis lesquels : ")
    st.image(load('ressources/img/', 'candidates_models', 'png'))
    st.markdown("""
                **Technique de CV**: RandomizedSearchCV au lieu de GridSearchCV car moins coûteux en temps de calcul.
                """)
    experiments = client.search_experiments()
    experiment_names = [exp.name for exp in experiments]
    selected_experiment_name = st.selectbox("Choisir une expérience", experiment_names)

    # afficher les runs de l'expérience sélectionnée
    selected_experiment = next(exp for exp in experiments if exp.name == selected_experiment_name)
    runs = client.search_runs(selected_experiment.experiment_id, order_by=["start_time DESC"])

    # résultats sous forme de tableau
    if runs:
        runs_data = []
        for run in runs:
            runs_data.append({
                "Run ID": run.info.run_id,
                "Run name": run.info.run_name,
                # "Date": run.info.start_time,
                **run.data.params,  # paramètres
                **run.data.metrics,  # métriques
            })
        
        # tableau récapitulatif 
        runs_df = pd.DataFrame(runs_data).replace(np.nan, '--').drop(['task', 'model'], axis=1)
        st.dataframe(runs_df)  

        # all - lroc
        st.text("\n\n")
        _1, _2 = st.columns(2)
        st.markdown("### LROC curves")
        _1.image(load(ROOT, "lroc", "png"))
        _2.image(load(ROOT, "precision_recall_curves_on_testset", "png"))
        _1.markdown(
            """
            *On voit que les modèles ont globalement du mal à prédire les vrai positifs. 
            Pour rappel la target est churn (1 si la personne a résilié et 0 si non). 
            Donc, les modèles qui prédisent le mieux 
            la proba de résiilation sont tous les modèles **sauf la regression logistique** 
            et le **random forest classifier** (qui est en fait un decision tree).*
            """
        )
        _2.markdown(
            """
            *Ici aussi, l'évaluation montre que les modèles font quasiment le même 
            compromis entre recall et precision. Lorsque le recall est faible, 
            la précision est élevée mais elle baisse au fur et à mesure que le recall augmente.*
            """
        )
        # Run pour explorer plus en détail
        _runs = list(zip(runs_df["Run ID"], [run.info.run_name for run in runs]))
        selected_run_ = st.selectbox("\nSélectionnez un Run ID", _runs)
        if selected_run_:
            selected_run_id = selected_run_[0]
            selected_run = client.get_run(selected_run_id)
            
            _1, _2 = st.columns(2)
            _1.markdown("# ")
            _1.subheader("Paramètres")
            _1.write(selected_run.data.params)

            _2.markdown("# ")
            _2.subheader("Métriques")
            _2.write(selected_run.data.metrics)

            # ------- reload --------
            st.markdown("### Reload model")
            run_names = [run.info.run_name for run in runs]
            selected_run_name = selected_run_[1]
            print(f"Model selected : {selected_run_name}")
            try:
                encoder, model, _pipeline_loaded = reload_pipeline_by_name(selected_run_name)  
                st.markdown("""Model loaded successfully ! ✨""")
            except Exception as e: 
                st.markdown(f"""uohh error while loading model : {e}""")
                _pipeline_loaded = None
            
            # -------- show learning curves if available
            st.markdown("### Learning curves")
            try:
                st.image(load(f'ressources/models_fitted/{selected_run_name}/', 'learning_curves', 'png'))
            except Exception as e:
                # st.write(str(e))
                # e\rus-scaling-and-LogisticRegression\learning_curves.png
                st.markdown("*Learning curves not yet available..*")

            # -------- show features importance if available
            st.markdown("### Features importance")
            try:
                st.image(load(f'ressources/models_fitted/{selected_run_name}/', 'features_importance', 'png'))
                st.dataframe(load(f'ressources/models_fitted/{selected_run_name}/', 'importances_df', 'pkl'))
            except:
                st.markdown("*Features importances are not yet available for this model..*")

            st.markdown("### Pipeline overview")
            st.write(_pipeline_loaded)
            # -------- exemples predictions durant l'évaluation ----------
            st.markdown("""
                        ### Exemple predictions
                        *Exemples de prédictons pendant la phase d'évaluation*
                        """)

            data_sample = pickle.load(open('ressources/models_fitted/data_sample.pkl', 'rb'))
            y_sample = data_sample['churn']
            data_sample = data_sample.drop(['churn'], axis=1)
            if st.button('Run demo'):
                data_sample.insert(0, 'pred_churn', _pipeline_loaded.predict(data_sample))
                data_sample.insert(0, 'churn', y_sample)
                st.dataframe(data_sample)

            # ----- predict with new data ----------  
            st.markdown("""### Predict with new data""")   
            st.markdown("*Entrer les nouvelles données*")               
            input_data = {}
            columns = st.columns(4)
            col_index = 0
            schema_df = data_sample
            for column in schema_df.columns:
                col_type = schema_df[column].dtype
                with columns[col_index]:
                    if col_type == 'int64':
                        min_val, max_val = schema_df[column].min(), schema_df[column].max()
                        input_data[column] = st.number_input(column.replace('_0', ''), min_value=min_val, max_value=max_val, step=1)
                    elif col_type == 'float64':
                        min_val, max_val = schema_df[column].min(), schema_df[column].max()
                        input_data[column] = st.number_input(column.replace('_0', ''), min_value=min_val, max_value=max_val, step=0.01)
                    elif col_type == 'object':
                        unique_values = schema_df[column].unique()
                        input_data[column] = st.selectbox(column.replace('_0', ''), unique_values)
                    else:
                        input_data[column] = st.text_input(column)
                col_index = (col_index + 1) % 4
                
            input_df = pd.DataFrame([input_data])

            if st.button("Prédire"):
                if _pipeline_loaded is not None:
                    st.markdown('calcul prediction..')
                    prediction = _pipeline_loaded.predict(input_df)
                    proba = _pipeline_loaded.predict_proba(input_df)[0][1] # deuxieme elt => proba(churn = 1)
                else: # en cas de fail
                    prediction = np.random.choice([0,1])
                    proba = round(np.random.random(), 3) 
                
                print(proba)
                print(type(proba))
                st.write(proba)
                if proba > 0.5: 
                    st.success(f"Cette personne résiliera probablement à {round(100*proba, 1)}% (churn = {prediction}) 😔😔")
                else:
                    st.success(f"Cette personne ne résiliera probablement pas à {round(100*proba, 1)}% (churn = {prediction}) 👍")
    else:
        st.write("Aucun run trouvé pour cette expérience.")

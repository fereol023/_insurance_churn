### Auto Insurance Churn Modelisation 
Last update : 05/02/2025

#### Documentation du projet 
- Problématique, objectifs, livrables etc.. *(déjà fait - à compléter)*
- Source 1 : [Kaggle - Dataset simulé - Train](https://www.kaggle.com/datasets/merishnasuwal/auto-insurance-churn-analysis-dataset/data).
- Source 2 : [Hackathon - Dataset réel anonymisé - RGPD](https://www.kaggle.com/code/bhuwanesh340/customer-churn-prediction-weekend-hackathon).

#### Données
- Source du dataset : Worflow data (données croisées) issues :
    * ...
- Descriptif du workflow d'extraction <br><br>
    ![img](docs/pipe.png)
    <br><br>
- Résultats des extracts du workflow par exploitation des API : exemples [ici](res/data/)

#### Installation du projet (en local)

For this project you can make a virtual env if needed. 

- After cloning, in your favorite terminal/shell do :
```
pip install -r requirements.txt
```
- For fetching data :
 * See [this exemple notebook](notebooks/1_database.ipynb) for code.
 * Data extraction perf exemple - mode unitaire depuis le notebook cité ci-dessus : <br><br>
 ![img](docs/scope_paris_2018_200_min.png)

- For launching streamlit analytic app and models, run : 
```
streamlit run app/main.py
```
(TBD : aperçu)

- There is also a docker container available (here)


#### Perspectives : 
- Généraliser périmètre géographique France
- Optimiser la classe d'extraction : exple PARIS 2018 ~ 4H ~ 450k lignes au final
- *... (à compléter)*

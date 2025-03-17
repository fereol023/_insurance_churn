from contents import *

text = """

**Contexte**

Les compagnies d'assurance évoluent dans un environnement très concurrentiel. Avec les diverses données collectées auprès de millions de clients, il est extrêmement difficile d'analyser et de comprendre les raisons qui poussent un client à changer de fournisseur d'assurance. Dans cette une industrie où l'acquisition et la fidélisation des clients sont tout aussi importantes, et où la première est un processus plus coûteux, les compagnies d'assurance s'appuient sur les données pour comprendre le comportement des clients afin de prévenir les résiliations. Ainsi, savoir à l'avance si un client est susceptible de changer de fournisseur permet aux compagnies d'assurance de mettre en place des stratégies pour éviter que cela ne se produise

*Ce projet utilise deu datasets différents. Un dataset issu d'une base de données simulé et un dataset autre avec des données réelles mais dont les variables sont nommées feature 1,2, 3... et où les données ont été transformées (transformation incoonue à priori).*

### Data simulée

- Source database relationnelle : https://www.kaggle.com/datasets/merishnasuwal/auto-insurance-churn-analysis-dataset/data?select=address.csv

Nous disposons initialement de 4 tables joignables avec des clées primaires/étrangères (ADDRESS_ID ou INDIVIDUAL_ID).

1. address.csv: contient les informations sur les addresses
2. customer.csv: contient les informations sur les clients (customer)
3. demographic.csv: contient des données demographiques
4. termination.csv: contient les données sur la fin des contrats

- *Info : Les addresses sont uniques mais plusieurs clients peuvent avoir les mêmes adresses et toutes les personnes n'ont pas leur informations démographiques disponibles (données manquantes)* 

Le dataset comporte plus de 2millions de clients et environs 1.5 millions d'adresses uniques. Avec le calcul de la variable target, on sait qu'environs 10% des personnes ont résilié leurs contrats.

- *Autre info des auteurs : Les données portent sur des clients d'une société d'assurance fictive localisée à **Dallas au Texas**.. donc les adresses sont fictives. Le dataset est utilisable pour la modélisation de la probabilité de résilisation et la logique sous-jacente utilisée pour la simulation de ce dataset s'aligne avec les études sur la prédiction de la résilisation dans le monde réel. **Donc on peut modéliser sur ce dataset et ré appliquer sur de vraies données.***


### Hackathon Data => real world data

- Source dataset : https://www.kaggle.com/datasets/k123vinod/insurance-churn-prediction-weekend-hackathon?resource=download

Nous disposons ici d'un dataset sur des features labelisées toujours sur le sujet de la modélisation. Ce dataset est basé sur des données réelles mais contient **les features sont renommées et transformées** dans le cadre d'un [!hackathon IA](https://www.machinehack.com/course/insurance-churn-prediction-weekend-hackathon-2/). L'objectif dudit hackathon était juste de construire un modèle de machine learning qui permettrait de prédire la probabilité de résiliation.

### Objectif et challenge

Nous allons utiliser le premier dataset de données simulées pour construire un modèle et ensuite tenter de le généraliser sur les données réelles du hackathon.
- Challenge 1 : A priori, on sait que les modèles de machine learning généralisent mal donc on s'attend à des performances pas super à la fin.
- Challenge 2 : Le dataset simulé contient des millions d'observations dans des tables éparses. Il faut reconstituer la base et ensuite la nettoyer mais on aura un problème de dataviz.
- Challenge 3 : Il faudra trouver une manière d'établir la correspondance entre les features du dataset 1 et celles du daatset 2 car elles n'ont pas le même nom ni forcément le même range de valeurs mais juste les mêmes distributions statistiques. A priori c'est possible, mais il faut trouver la bonne technique.

"""

def main():

    st.title("Prédire la probabilité de résiliation d'un contrat d'assurance 📝")
    st.markdown(
        body = text
    )
    
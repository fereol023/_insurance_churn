import re, pickle, os
import numpy as np, pandas as pd
from functools import lru_cache
from unidecode import unidecode
from datetime import datetime

def load_pickle(fpath, is_optional=False):
    if (not os.path.exists(fpath) and (not is_optional)):
        raise Exception(f"File {fpath} does not exist and is not optional !")
    with open(fpath, 'rb') as f:
        res = pickle.load(f)
    return res


def save_pickle(obj, fpath):
    """obj : serialisable obj"""
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Sauvegarde ok at : {fpath}")


def get_entropy(pk, L):
    # pk : probabilité de chaque valeur
    # L : nombre de valeurs possibles
    # d : dataframe
    # cols : liste des colonnes à traiter
    op = - (pk * np.log(pk) / np.log(L)).sum()
    return op
 

def compute_entropies(d, cols):
    print('Computing entropies..')
    entropies = []
    for i,col in enumerate(cols):
        pk = d[col].value_counts(normalize=True, dropna = False).values
        entropy = round(get_entropy(pk, len(d)),2)
        entropies.append({"col": col, "entropy": entropy})
    return entropies      


def get_cross_entropy(pk, qk):
    """
    Calcule l'entropie croisée entre deux distributions de probabilité pk et qk.
    
    :param pk: np.array, distribution réelle (P, par ex. probabilité des classes).
    :param qk: np.array, distribution prédite (Q, estimée par un modèle).
    :return: float, valeur de l'entropie croisée.
    """
    epsilon = 1e-15
    qk = np.clip(qk, epsilon, 1 - epsilon)  # Évite log(0)
    return -np.sum(pk * np.log(qk))

def compute_cross_entropies(d, col_pairs):
    """
    Calcule les entropies croisées pour des paires de colonnes dans un dataframe.
    
    :param d: pd.DataFrame, dataframe contenant les données.
    :param col_pairs: list of tuples, chaque tuple (col_P, col_Q) représente une paire 
                      de colonnes pour lesquelles calculer l'entropie croisée.
    :return: list, entropies croisées par paire de colonnes.
    """
    print('Computing cross-entropies...')
    cross_entropies = []
    
    for col_P, col_Q in col_pairs:

        # Alignement des distributions (au cas où les valeurs diffèrent)
        unique_values = sorted(set(d[col_P].dropna().unique()) | set(d[col_Q].dropna().unique()))
        pk_full = np.array([d[col_P].value_counts(normalize=True, dropna=False).get(v, 0) for v in unique_values])
        qk_full = np.array([d[col_Q].value_counts(normalize=True, dropna=False).get(v, 0) for v in unique_values])

        # Calcul de l'entropie croisée
        cross_entropy = round(get_cross_entropy(pk_full, qk_full), 4)
        cross_entropies.append({"col_pair": (col_P, col_Q), "cross_entropy": cross_entropy})
    
    return cross_entropies

def compute_insights(data):
    """Afficher en une fois le taux de na, de nulls et l'entropie."""
    _ = pd.DataFrame(data.isnull().sum(), columns=['null freq'])/data.shape[0]
    _['na freq'] = data.isna().sum()/data.shape[0]
    _['entropy'] = pd.DataFrame(compute_entropies(data, data.columns)).set_index('col')['entropy']
    _ = round(_, 3)
    return _.sort_values(by='entropy')

@lru_cache(maxsize=1024)
def normalize_name(colname):
    pat1, pat2 = re.compile('[^0-9a-zA-Z]+'), re.compile('_+')
    return pat1.sub('_', pat2.sub('_', colname))


def normalize_df_colnames(df):
    return df.rename(columns={c: normalize_name(unidecode(c)).lower() for c in df.columns})


def get_today_date():
    return datetime.today().strftime('%Y_%m_%d')

def load_parquet_data(_PATH):
    print(f"Loading parquet data from : {_PATH}..")
    return pd.read_parquet(_PATH)
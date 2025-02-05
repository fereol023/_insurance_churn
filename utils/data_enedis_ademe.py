import pandas as pd, numpy as np, requests, logging

class DataEnedisAdeme:
    # enedis records endpoint has limitations ~ default 10 if not set  # combier limit et offset pour imiter la pagination limite à 10_000 lignes 
    # while export endpoint has no limitations of rows
    def __init__(self, path_enedis_csv=None):
        self.path_enedis_csv = path_enedis_csv
        self.debugger = {}

    def load_enedis_from_csv(self):
        res = pd.read_csv(self.path_enedis_csv, sep=';')
        res = res.rename(columns={'Adresse': 'adresse', 'Nom Commune': 'nom_commune', 'Code Commune': 'code_commune'})
        return res

    def load_get_data_pandas(self, url):
        """Extract pandas dataframe from any valid url."""
        res = requests.get(url).json().get('results')
        return pd.DataFrame(res)

    def get_url_enedis_year(self, annee): 
        """Génerer l'url pour requeter l'api enedis. filtres sur le nbre de lignes.
        param: annee : int année souhaitée
        """
        return f"https://data.enedis.fr/api/explore/v2.1/catalog/datasets/consommation-annuelle-residentielle-par-adresse/records?where=annee%20%3D%20date'{annee}'"

    def get_url_enedis_year_rows(self, annee, rows): 
        """Génerer l'url pour requeter l'api enedis. filtres sur l'année et le nbre de lignes.
        param: annee : int année souhaitée
               rows : nbre de lignes souhaitées.
        """
        return f"https://data.enedis.fr/api/explore/v2.1/catalog/datasets/consommation-annuelle-residentielle-par-adresse/records?where=annee%20%3D%20date'{annee}'&limit={rows}"

    def get_url_ademe_filter_on_ban(self, key):
        """Générer l'url pour requeter l'api de l'ademe (dpe depuis juillet 2021)."""
        return f"https://data.ademe.fr/data-fair/api/v1/datasets/dpe-v2-logements-existants/lines?size=1000&format=json&qs=Identifiant__BAN%3A{key}"

    def get_ban_res(self, addr):
        """requeter l'api de ban de facon unitaire càd une seule adresse en entrée.
           produit une seule sortie avec les infos sur l'adresse si elle existe (dont l'id BAN).
           Si l'adresse n'existe pas renvoie None.
        """
        ADDOK_URL = 'http://api-adresse.data.gouv.fr/search/'

        params = {
            'q': addr,
            'limit': 1
        }
        response = requests.get(ADDOK_URL, params=params)
        if response.status_code == 200:
            j = response.json()
            if len(j.get('features')) > 0:
                first_result = j.get('features')[0]
                lon, lat = first_result.get('geometry').get('coordinates')
                first_result_all_infos = { **first_result.get('properties'), **{"lon": lon, "lat": lat}, **{'full_adress': addr}}
                return first_result_all_infos
            else:
                return
        else:
            return

    def get_enedis_with_ban_pandas(self, requete_url_enedis, from_export): 
        """obtenir le df pandas des données enedis (requete) + les données de la BAN pour les adresses trouvées."""
        
        if not from_export:
            # 1- requeter enedis
            logging.warning(f"Extract from url enedis :\n {requete_url_enedis}")
            self.debugger.update({'source_enedis': requete_url_enedis})

            enedis_data = self.load_get_data_pandas(requete_url_enedis)
        else:
            # 1bis- lire depuis csv enedis
            logging.warning(f"Loading from :\n {self.path_enedis_csv}")
            self.debugger.update({'source_enedis': self.path_enedis_csv})

            enedis_data = self.load_enedis_from_csv()
            enedis_data = enedis_data[enedis_data['Code Département']==75] # restriction paris 
        self.debugger.update({'sample_enedis_data': enedis_data.tail(5)})

        # 2- extraire les adresses complètes
        enedis_adresses_list = list(zip(enedis_data.adresse.values.tolist(),
                                        enedis_data.code_commune.values.tolist(),
                                        enedis_data.nom_commune.values.tolist()))
        enedis_adresses_list = [f"{a} {b} {c}" for a,b,c in enedis_adresses_list]
        enedis_data['full_adress'] = enedis_adresses_list
        self.debugger.update({'enedis_full_adresses': set(enedis_adresses_list)})

        # 3- requeter l'api de la BAN sur les adresses enedis
        # ban_data = pd.DataFrame([self.get_ban_res(_) for _ in enedis_adresses_list]) # drop nones 
        # ban_data = [self.get_ban_res(_) for _ in enedis_adresses_list]
        sample_enedis = enedis_adresses_list #[0:2005]
        ban_data = [self.get_ban_res(_) for _ in sample_enedis]
        ban_data = [_ for _ in ban_data if str(_).lower() != 'none']
        logging.warning(f"Valid data BAN : {len(ban_data)}/{len(sample_enedis)}")
        ban_data = pd.DataFrame(ban_data)
        vectorized_upper = np.vectorize(str.upper, cache=True)
        ban_data['label'] = vectorized_upper(ban_data['label'].values)
        self.debugger.update({'sample_ban_data': ban_data.tail(5)})

        # 4- renvoyer le resultat mergé : si une adresse est pas  trouvée on tej la data enedis (cf. inner join)
        return pd.merge(enedis_data, ban_data, how='inner', left_on='full_adress', right_on='full_adress').rename(columns={'id': 'id_BAN'})

    def get_enedis_with_ban_with_ademe(self, requete_url_enedis, from_export:bool=False):
        """extraire la data enedis x ban x ademe au complet à partir d'un dataframe enedis."""
        
        # extraire enedis avec les infos de la BAN
        enedis_with_ban_data = self.get_enedis_with_ban_pandas(requete_url_enedis, from_export=from_export)
        enedis_with_ban_data.to_csv('../ressources/data/0_archive/enedis_with_ban_data_tmp.csv') 
        enedis_with_ban_data_id_BAN_list = enedis_with_ban_data.id_BAN.values.tolist()
        del enedis_with_ban_data ##SUPPR LE TEMPS DE REQUETER ADEME // MEMEORY ERROR HANDLING
        # sur la base des Identifiants BAN de enedis, aller chercher les logements mappés sur ces codes BAN
        # plusieurs adresses possibles pour un id BAN 
        # car les données enedis sont regroupées/agrégées. Toutefois, on dispose de la moyenne.
        ademe_data = []
        ## TODO: MEMORY ERROR HANDLING
        # ademe_data_res = [requests.get(self.get_url_ademe_filter_on_ban(_)).json().get('results') for _ in enedis_with_ban_data.id_BAN.values.tolist()] # liste à 2 niveaux # pour chaque Id_BAN on a plusieurs lignes ademe
        ademe_data_res = [requests.get(self.get_url_ademe_filter_on_ban(_)).json().get('results') for _ in enedis_with_ban_data_id_BAN_list] # liste à 2 niveaux # pour chaque Id_BAN on a plusieurs lignes ademe        
        for _ in ademe_data_res:
            ademe_data.extend(_)
        del ademe_data_res
        ademe_data = pd.DataFrame(ademe_data)
        ademe_data = ademe_data.add_suffix('_ademe')
        ademe_data.to_csv('../ressources/data/0_archive/ademe_data_tmp.csv')

        # RELOAD LE TMP FILE ENEDIS DATA
        enedis_with_ban_data = pd.read_csv('../ressources/data/0_archive/enedis_with_ban_data_tmp.csv') 
        enedis_with_ban_data = enedis_with_ban_data.add_suffix('_enedis_with_ban')

        return pd.merge(ademe_data,
                        enedis_with_ban_data,
                        how='left',
                        left_on='Identifiant__BAN_ademe',
                        right_on='id_BAN_enedis_with_ban')

    def extract_year_rows(self, year:int=2018, rows:int=20):
        res = self.get_enedis_with_ban_with_ademe(self.get_url_enedis_year_rows(year,rows))
        logging.warning(f"Extraction results : {res.shape[0]} rows, {res.shape[1]} columns.")
        return res
    
    def extract(self):
        res = self.get_enedis_with_ban_with_ademe(None, from_export=True)
        logging.warning(f"Extraction results : {res.shape[0]} rows, {res.shape[1]} columns.")
        self.result = res
        return res.head(3)
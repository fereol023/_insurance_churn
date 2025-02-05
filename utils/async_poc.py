import aiohttp
import asyncio

class BANRequester:
    def __init__(self):
        self.ADDOK_URL = 'http://api-adresse.data.gouv.fr/search/'

    async def get_ban_res(self, session, addr):
        """
        Requêter l'API de BAN pour une seule adresse en utilisant aiohttp.
        """
        params = {
            'q': addr,
            'limit': 1
        }
        try:
            async with session.get(self.ADDOK_URL, params=params) as response:
                if response.status == 200:
                    print(f'{addr}')
                    data = await response.json()
                    if len(data.get('features', [])) > 0:
                        first_result = data['features'][0]
                        lon, lat = first_result['geometry']['coordinates']
                        return {
                            **first_result['properties'],
                            "lon": lon,
                            "lat": lat,
                            'full_adress': addr
                        }
        except Exception as e:
            print(f"Error fetching address {addr}: {e}")
        return None

    async def fetch_all_addresses(self, addresses, max_concurrent_requests):
        """
        Gérer la requête de 400 000 adresses de manière optimisée.
        """
        timeout = aiohttp.ClientTimeout(
                total=6000,  # Temps total maximum pour une requête
                connect=1000,  # Délai pour établir la connexion
                sock_read=1000,  # Délai pour lire la réponse
                sock_connect=1000  # Délai pour la connexion du socket
            )
        connector = aiohttp.TCPConnector(limit=max_concurrent_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.get_ban_res(session, addr) for addr in addresses]
            results = await asyncio.gather(*tasks)
        return results


# Exemple d'utilisation
if __name__=='__main__':
    
    requester = BANRequester()

    import pandas as pd, datetime
    tmp_adresses = list(pd.read_csv('ressources/data/enedis_2018_full_adresses.csv').full_adress.values)

    # Exécuter l'opération principale
    async def main():
        results = await requester.fetch_all_addresses(tmp_adresses[0:11], max_concurrent_requests=1)
        # Filtrer les résultats valides
        valid_results = [res for res in results if res is not None]
        print(f"Nombre d'adresses valides: {len(valid_results)}")
        df = pd.DataFrame(valid_results)
        
        # Enregistrer dans un fichier CSV
        csv_filename = "ressources/data/enedis_with_ban.csv"
        df.to_csv(csv_filename, index=False)

    d = datetime.datetime.now()
    asyncio.run(main())
    print(f"Durée : {datetime.datetime.now() - d}")

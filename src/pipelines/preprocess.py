import sys
sys.path.append('.')

from utils.fonctions import get_today_date, load_parquet_data
from src.artifacts.data_processing import DataPreprocessor

INPUT_PATH = 'ressources/data/2_intermediary/enedis_ban_ademe_extract_PARIS_2022.parquet'
OUTPUT_PATH = f'ressources/data/3_cleaned/{get_today_date()}/enedis_ban_ademe_extract_PARIS_2022.parquet'

if __name__=='__main__':

    df = load_parquet_data(INPUT_PATH)
    pipeline = DataPreprocessor(df)
    pipeline.run().save(to_path=OUTPUT_PATH)

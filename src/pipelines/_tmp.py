import sys
sys.path.append('.')

from utils.fonctions import get_today_date, load_parquet_data
from src.artifacts.data_processing import TestExp


if __name__=='__main__':

    pipeline = TestExp(f"Exêrience_test_{get_today_date()}_01")
    pipeline.run()

    path = ""

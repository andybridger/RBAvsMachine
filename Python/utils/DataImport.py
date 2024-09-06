import numpy as np
import pandas as pd
from pathlib import Path

class DataImport:  # Read data
    def __init__(self):
        pass

    @staticmethod
    def read_monthly(form: str = "pd"):
        # link = "https://github.com/andybridger/RBAvsMachine/raw/master/df_data_monthly.csv"
        link = Path('../../data/raw/df_data_monthly.csv')
        if form == "pl":
            return pl.read_csv(link)
        elif form == "pd":
            df_temp = pd.read_csv(link)
            df_temp["date"] = pd.to_datetime(df_temp["date"])
            return df_temp
        else:
            raise TypeError("Should be of type 'pl' or 'pd'")

    @staticmethod
    def read_quarterly(form: str = "pd"):
        # link = "https://github.com/andybridger/RBAvsMachine/raw/master/df_data_quarterly.csv"
        link = Path('../../data/raw/df_data_quarterly.csv')
        if form == "pl":
            return pl.read_csv(link)
        elif form == "pd":
            df_temp = pd.read_csv(link)
            df_temp["date"] = pd.to_datetime(df_temp["date"])
            return df_temp
        else:
            raise TypeError("Should be of type 'pl' or 'pd'")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


file_dir = "/Users/tianyudu/Downloads/claims_final.csv"

def read_file(
        d: str) -> pd.DataFrame:
    df = pd.read_csv(d, header=None)
    col_names = ["fam_id", "fam_mem_id", "pro_id", "pro_type", "state", "date", "med_proc_code", "amt"]
    df.columns = col_names
    return df

def read_cleaned_data(
    d: str) -> pd.DataFrame:
    df = pd.read_csv(d, header=0)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

def normalize_data(df: pd.DataFrame) \
    -> (np.ndarray, StandardScaler):
    values = df.values
    scaler = StandardScaler().fit(values)
    values = scaler.transform(values)
    return (values, scaler)

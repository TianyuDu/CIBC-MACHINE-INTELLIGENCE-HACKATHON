import pandas as pd
import numpy as np

def read_file(
        d: str="https://s3.amazonaws.com/cibchack/data/claims_final.csv") -> pd.DataFrame:
    df = pd.read_csv(d, header=None)
    col_names = ["fam_id", "fam_mem_id", "pro_id", "pro_type", "state", "date", "med_proc_code", "amt"]
    df.columns = col_names
    return df
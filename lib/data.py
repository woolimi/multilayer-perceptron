import pandas as pd
from lib.print import danger, warning

def load_csv(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename, header=None)
    except FileNotFoundError:
        print(f"{danger('Error: ')}{danger('File not found')}")
    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize only the numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    return df

import pandas as pd


def data_processing(df: pd.DataFrame) -> pd.DataFrame:
    print(warning("Data processing..."))

    df.drop(columns=[0], inplace=True)
    standardize(df)
    df[1] = df[1].map({"M": 1, "B": 0})

    y = df[1].values
    X = df.drop(columns=df.columns[0], axis=1).values

    return X, y

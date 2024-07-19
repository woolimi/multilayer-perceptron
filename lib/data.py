import pandas as pd

def load_csv(path: str):
    df = pd.read_csv(path, header=None)
    y = df[0].values
    X = df.drop(columns=df.columns[0], axis=1).values
    return X, y

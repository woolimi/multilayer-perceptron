import pandas as pd
import lib.print as danger

def load_csv(filename: str) -> pd.DataFrame:
    try:

        df = pd.read_csv(filename, header=None)
    except:
        print(f"{danger('Error: ')}{danger('File not found')}")
    return df

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    # Separate numeric
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Standardize only the numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    return df
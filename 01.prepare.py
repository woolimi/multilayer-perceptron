from lib.print import success, warning, danger
import pandas as pd


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

if __name__ == "__main__":
    """
    Program to load dataset and separate into train and test
    """
    
    df = load_csv("./data.csv")

    print(warning("Removing first column..."))
    df.drop(columns=[0], inplace=True)

    print(warning("Standardizing..."))
    standardize(df)

    print(warning("Mapping M and B to 1 and 0..."))
    df[1] = df[1].map({"M": 1, "B": 0})

    print(warning("Creating train and test datasets..."))
    df_train = df.sample(frac=0.8, random_state=42)
    df_validate = df.drop(df_train.index)

    # Test dataset
    df_test = df

    # Create train and test files
    df_train.to_csv("./train.csv", index=False, header=False)
    df_validate.to_csv("./validate.csv", index=False, header=False)
    df_test.to_csv("./test.csv", index=False, header=False)
    print(success(f"Successfully created dataset ✨"))
    print(f"Train dataset {df_train.shape} created at ./train.csv")
    print(f"Validate dataset {df_validate.shape} created at ./validate.csv")
    print(f"Test dataset {df_test.shape} created at ./test.csv")

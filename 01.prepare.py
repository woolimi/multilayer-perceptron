from lib.mlp import load_csv, standardize
from lib.print import success, warning
import pandas as pd


# Program to load dataset and separate into train and test
if __name__ == "__main__":
    df = load_csv("./data.csv")

    print(warning("Removing first column..."))
    df.drop(columns=[0], inplace=True)

    print(warning("Standardizing..."))
    standardize(df)

    print(warning("Mapping M and B to 1 and 0..."))
    df[1] = df[1].map({"M": 1, "B": 0})

    print(warning("Creating train and test datasets..."))
    df_train = df.sample(frac=0.5, random_state=42)

    # Test dataset
    df_test = df

    # Create train and test files
    df_train.to_csv("./train.csv", index=False, header=False)
    df_test.to_csv("./test.csv", index=False, header=False)
    print(success(f"Successfully created dataset ✨"))
    print(f"Train dataset {df_train.shape} created at ./train.csv")
    print(f"Test dataset {df_test.shape} created at ./test.csv")

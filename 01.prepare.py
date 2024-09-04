from lib.print import success, warning, danger
from lib.data import load_csv


if __name__ == "__main__":
    """
    Program to load dataset and separate into train and test
    """
    
    df = load_csv("./data.csv")

    print(warning("Creating train and test datasets..."))
    df_train = df.sample(frac=0.50)
    df_validate = df.drop(df_train.index)

    # Create train and test files
    df_train.to_csv("./data_training.csv", index=False, header=False)
    df_validate.to_csv("./data_test.csv", index=False, header=False)

    print(success(f"Successfully created dataset ✨"))
    print(f"Train dataset {df_train.shape} created at ./data_training.csv")
    print(f"Validate dataset {df_validate.shape} created at ./data_test.csv")

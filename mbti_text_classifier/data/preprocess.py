import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, target_column, train_size, val_size, test_size, random_state):
    stratify_col = df[target_column]

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    val_size_adjusted = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_df[target_column],
    )

    return train_df, val_df, test_df


def preprocess_mbti_dataset(
    data_path, target_column, train_size, val_size, test_size, random_state
):
    df = pd.read_csv(data_path)
    return split_data(df, target_column, train_size, val_size, test_size, random_state)

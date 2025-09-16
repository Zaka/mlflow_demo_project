import os
from sklearn.datasets import fetch_openml
import pandas as pd

adult = fetch_openml("adult", version=2)
df = adult.frame

# Create the output class

# Output class is 'class' column
df["class"] = df["class"].map({"<=50K": 0, ">50K": 1})

# Create one hot encoding for workclass

ohe_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def create_ohe_columns(df, columns):
    for column in columns:
        ohe = pd.get_dummies(df[column], prefix=column, drop_first=True, dummy_na=True)

        if isinstance(ohe.columns, pd.CategoricalIndex):
            ohe.columns = ohe.columns.astype(object)  # becomes array of Python objects
            ohe.columns = pd.Index(ohe.columns)  # plain Index

        # Replace NaN column label(s)
        ohe.columns = ohe.columns.where(~ohe.columns.isna(), column + "missing")

        # (optional) ensure they're strings
        ohe.columns = pd.Index(map(str, ohe.columns))

        # Join OHE back with the rest of the columns
        df = df.join(ohe, how="left")

    # Remove all categorical columns
    new_columns = [col for col in df.columns if col not in columns]
    df = df[new_columns]

    return df


df = create_ohe_columns(df, ohe_columns)

new_columns = [col for col in df.columns if col != "education-num"]
df = df[new_columns]

# Store pickled dataset

df.to_pickle(os.path.join('.', 'data', 'processed', 'dataset.pkl'))

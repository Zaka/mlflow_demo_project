import os
from sklearn.datasets import fetch_openml

def load_dataset():
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame

    # Create the output class


    # Output class is 'class' column
    df["class"] = df["class"].map({"<=50K": 0, ">50K": 1})

    # Drop education-num because it is a numerical representation of education.
    new_columns = [col for col in df.columns if col != "education-num"]
    df = df[new_columns]

    # Store pickled dataset
    
    try:
        path = os.path.join('.', 'data', 'processed', 'dataset.pkl')
        df.to_pickle(path)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path))
        df.to_pickle(path)

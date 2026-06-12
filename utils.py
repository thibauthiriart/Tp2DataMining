import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data
def load_data(filename: str | None = None) -> pd.DataFrame:

    if filename is None:
        csvs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {DATA_DIR}")
        filename = csvs[0]
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


@st.cache_data
def prepare_clients(
    filename: str | None = None,
    k: int = 5,
    iqr_k: float = 1.5,
    cols_winsor: tuple[str, ...] = ("revenu", "solde_compte"),
) -> pd.DataFrame:
    df = load_data(filename).copy()

    id_col = next((c for c in df.columns if c.lower() == "id"), None)
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != id_col
    ]
    cat_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and c != id_col
    ]
    if num_cols:
        imputer = KNNImputer(n_neighbors=k)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    for c in cols_winsor:
        if c in num_cols:
            q1, q3 = np.percentile(df[c], [25, 75])
            iqr = q3 - q1
            low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            df[c] = np.clip(df[c], low, high)
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


@st.cache_data
def load_iris_df() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    X = iris.data
    df = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns, index=X.index
    )
    df["species"] = iris.target_names[iris.target]
    return df


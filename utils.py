import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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


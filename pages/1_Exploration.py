import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

st.set_page_config(page_title="Exploration", layout="wide")
st.title("Exploration des données")

try:
    df = load_data()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

st.subheader("Aperçu")
st.dataframe(df.head())

c1, c2, c3 = st.columns(3)
c1.metric("Lignes", df.shape[0])
c2.metric("Colonnes", df.shape[1])
c3.metric("Valeurs manquantes", int(df.isna().sum().sum()))

ORDINAL_CATEGORIES = {
    "categorie": ["Standard", "Premium"],
}


def detect_type(serie: pd.Series) -> str:
    s = serie.dropna()
    name = serie.name

    if name in ORDINAL_CATEGORIES:
        return "qualitatif ordinal"

    if not pd.api.types.is_numeric_dtype(s):
        return "qualitatif nominal"
    if pd.api.types.is_integer_dtype(s) or np.all(s == s.astype(int)):
        return "quantitatif discret"
    return "quantitatif continu"


colonnes = [c for c in df.columns if c.lower() != "id"]
variable = st.selectbox("Choisissez une variable", colonnes)

serie = df[variable]
var_type = detect_type(serie)
st.markdown(f"**Type détecté :** `{var_type}`")

st.subheader("Statistiques")
if var_type.startswith("quantitatif"):
    st.dataframe(serie.describe().to_frame())
else:
    st.dataframe(serie.value_counts(dropna=False).rename("effectif"))

st.subheader("Visualisation")
if var_type.startswith("quantitatif"):
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=(8, 5),
    )
    sns.boxplot(x=serie.dropna(), ax=ax_box, color="skyblue")
    ax_box.set(xlabel="")
    sns.histplot(serie.dropna(), kde=True, ax=ax_hist, color="steelblue")
    ax_hist.set_xlabel(variable)
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = serie.value_counts(dropna=False)
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="steelblue")
    ax.set_xlabel(variable)
    ax.set_ylabel("Fréquence")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

st.subheader("Récapitulatif des variables")
recap = pd.DataFrame(
    {
        "variable": colonnes,
        "type": [detect_type(df[c]) for c in colonnes],
        "n_modalites": [df[c].nunique(dropna=True) for c in colonnes],
    }
)
st.dataframe(recap, use_container_width=True)

qual_nom = recap.loc[recap["type"] == "qualitatif nominal", "variable"].tolist()
qual_ord = recap.loc[recap["type"] == "qualitatif ordinal", "variable"].tolist()
quant_dis = recap.loc[recap["type"] == "quantitatif discret", "variable"].tolist()
quant_con = recap.loc[recap["type"] == "quantitatif continu", "variable"].tolist()

st.subheader("Synthèse")
st.markdown(
    f"""
- **Attributs qualitatifs nominaux** : {", ".join(qual_nom) if qual_nom else "—"}
- **Attributs qualitatifs ordinaux** : {", ".join(qual_ord) if qual_ord else "—"}
- **Attributs quantitatifs discrets** : {", ".join(quant_dis) if quant_dis else "—"}
- **Attributs quantitatifs continus** : {", ".join(quant_con) if quant_con else "—"}
"""
)

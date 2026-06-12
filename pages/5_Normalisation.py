import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from utils import load_data

st.set_page_config(page_title="Normalisation", layout="wide")
st.title("Normalisation")

try:
    df = load_data()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()


quant_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c]) and c.lower() != "id"
]
id_col = next((c for c in df.columns if c.lower() == "id"), None)
ids = df[id_col] if id_col is not None else pd.Series(df.index, name="id")


defaut = [c for c in ("revenu", "age", "depenses_mensuelles") if c in quant_cols]
variables = st.multiselect(
    "Variables quantitatives à normaliser",
    quant_cols,
    default=defaut or quant_cols[:3],
)
if not variables:
    st.info("Sélectionnez au moins une variable.")
    st.stop()




X_raw = df[variables].copy()
X_raw = X_raw.fillna(X_raw.median(numeric_only=True))



scalers = {
    "Min-Max (MinMaxScaler)": MinMaxScaler(),
    "Z-score (StandardScaler)": StandardScaler(),
    "Robuste (RobustScaler)": RobustScaler(),
}
formules = {
    "Min-Max (MinMaxScaler)": r"x' = \dfrac{x - \min(x)}{\max(x) - \min(x)} \in [0, 1]",
    "Z-score (StandardScaler)": r"x' = \dfrac{x - \bar{x}}{\sigma}",
    "Robuste (RobustScaler)": r"x' = \dfrac{x - \mathrm{m\acute{e}diane}(x)}{\mathrm{IQR}(x)}",
}



st.subheader("Méthode de normalisation")
choix = st.radio("Méthode", list(scalers.keys()), horizontal=True)
st.latex(formules[choix])

scaler = scalers[choix]
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_raw), columns=variables, index=X_raw.index
)



st.subheader("Distributions avant / après normalisation")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Avant** (échelles brutes)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=X_raw, orient="h", ax=ax, color="steelblue")
    ax.set_xlabel("valeur brute")
    st.pyplot(fig)
with col_b:
    st.markdown(f"**Après** ({choix})")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=X_scaled, orient="h", ax=ax, color="seagreen")
    ax.set_xlabel("valeur normalisée")
    st.pyplot(fig)

st.caption(
    "Sur des échelles brutes, les boxplots ne sont pas comparables : `revenu` ou "
    "`solde_compte` écrasent `nb_enfants` ou `satisfaction`. Après normalisation, "
    "toutes les variables partagent une échelle homogène."
)

with st.expander("Histogrammes superposés (avant / après)"):
    h1, h2 = st.columns(2)
    with h1:
        fig, ax = plt.subplots(figsize=(5, 4))
        for c in variables:
            ax.hist(X_raw[c], bins=25, alpha=0.5, label=c)
        ax.set_title("Avant")
        ax.legend(fontsize=7)
        st.pyplot(fig)
    with h2:
        fig, ax = plt.subplots(figsize=(5, 4))
        for c in variables:
            ax.hist(X_scaled[c], bins=25, alpha=0.5, label=c)
        ax.set_title("Après")
        ax.legend(fontsize=7)
        st.pyplot(fig)

st.divider()



st.subheader("Distance entre deux clients")
id_list = ids.tolist()
id_min, id_max = int(min(id_list)), int(max(id_list))

c1, c2 = st.columns(2)
id_a = c1.number_input(
    "Client A (id)", min_value=id_min, max_value=id_max, value=id_min, step=1
)
id_b = c2.number_input(
    "Client B (id)", min_value=id_min, max_value=id_max, value=min(id_min + 1, id_max), step=1
)

pos_a = ids.index[ids == id_a]
pos_b = ids.index[ids == id_b]
if len(pos_a) == 0 or len(pos_b) == 0:
    st.warning("Un des identifiants saisis n'existe pas dans le jeu de données.")
    st.stop()
pos_a, pos_b = pos_a[0], pos_b[0]



va_raw = X_raw.loc[pos_a].to_numpy(dtype=float)
vb_raw = X_raw.loc[pos_b].to_numpy(dtype=float)
va_sc = X_scaled.loc[pos_a].to_numpy(dtype=float)
vb_sc = X_scaled.loc[pos_b].to_numpy(dtype=float)

dist_raw = float(np.sqrt(np.sum((va_raw - vb_raw) ** 2)))
dist_sc = float(np.sqrt(np.sum((va_sc - vb_sc) ** 2)))

d1, d2 = st.columns(2)
d1.metric("Distance euclidienne (brute)", f"{dist_raw:.3f}")
d2.metric("Distance euclidienne (normalisée)", f"{dist_sc:.3f}")



contrib_raw = (va_raw - vb_raw) ** 2
contrib_sc = (va_sc - vb_sc) ** 2
detail = pd.DataFrame(
    {
        "variable": variables,
        f"client {id_a} (brut)": va_raw,
        f"client {id_b} (brut)": vb_raw,
        "écart²  (brut)": np.round(contrib_raw, 3),
        "part %  (brut)": np.round(100 * contrib_raw / contrib_raw.sum(), 1)
        if contrib_raw.sum() > 0 else 0.0,
        "part %  (norm.)": np.round(100 * contrib_sc / contrib_sc.sum(), 1)
        if contrib_sc.sum() > 0 else 0.0,
    }
)
st.dataframe(detail, use_container_width=True, hide_index=True)

dominante = variables[int(np.argmax(contrib_raw))] if contrib_raw.sum() > 0 else "—"
st.info(
    f"**Commentaire automatique** — Sur les données **brutes**, la distance vaut "
    f"`{dist_raw:.2f}` et est dominée à **{100 * contrib_raw.max() / contrib_raw.sum():.0f} %** "
    f"par la variable `{dominante}`, simplement parce qu'elle s'exprime dans les plus grands "
    f"nombres. Après **{choix}**, chaque variable pèse d'un poids comparable : la distance "
    f"`{dist_sc:.2f}` reflète une **dissimilarité réelle** et non l'unité de mesure."
)

st.divider()



st.subheader("9. Quelle variable domine la distance sur les données brutes ?")
ecarts = (X_raw.max() - X_raw.min()).sort_values(ascending=False)
dom_globale = ecarts.index[0]
st.markdown(
    f"""
La distance euclidienne **somme des carrés d'écarts** : une variable dont l'**amplitude**
(et donc l'écart entre deux clients) se chiffre en milliers domine mécaniquement celles qui
varient sur quelques unités.

Ici la variable de plus grande étendue est **`{dom_globale}`** (amplitude ≈ {ecarts.iloc[0]:,.0f}),
loin devant des variables comme `nb_enfants` ou `satisfaction`.

> **Pourquoi ?** L'euclidien est **sensible aux échelles** : sans normalisation, ce sont les
> **unités de mesure** (euros vs nombre d'enfants) qui décident de la distance, pas la
> structure des données. Normaliser remet toutes les variables sur un pied d'égalité.
"""
)




st.subheader("10. Normalisation en présence d'outliers")
cible = "revenu" if "revenu" in quant_cols else variables[0]
r = df[[cible]].dropna()

comp = pd.DataFrame(
    {
        "Min-Max": MinMaxScaler().fit_transform(r).ravel(),
        "Z-score": StandardScaler().fit_transform(r).ravel(),
        "Robuste": RobustScaler().fit_transform(r).ravel(),
    }
)
resume = comp.agg(["min", "max", "std"]).T.round(3)
resume.columns = ["min", "max", "écart-type"]

cc1, cc2 = st.columns([1, 1])
with cc1:
    st.markdown(f"Étalement de **`{cible}`** après chaque méthode :")
    st.dataframe(resume, use_container_width=True)
with cc2:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=comp, ax=ax, palette="Set2")
    ax.set_ylabel("valeur normalisée")
    ax.set_title(f"`{cible}` normalisé")
    st.pyplot(fig)

st.markdown(
    """
**Méthode à préférer : `RobustScaler`.**

- `MinMaxScaler` utilise `min` et `max` : un **seul outvalue extrême** fixe le maximum, et
  **tasse toutes les autres observations** près de 0 — l'information utile est écrasée.
- `StandardScaler` utilise la moyenne et l'écart-type, **gonflés par les valeurs extrêmes** :
  les scores restent comprimés et la « moyenne 0 / écart-type 1 » est trompeuse.
- `RobustScaler` centre sur la **médiane** et divise par l'**IQR**, deux statistiques
  **robustes** : l'étalement du gros de la distribution est préservé et les outliers ne
  déforment pas l'échelle (ils se contentent de prendre de grandes valeurs normalisées).

> Sur `revenu` (variable étalée à droite avec de gros montants), comparez ci-dessus : Min-Max
> concentre la masse vers 0, tandis que la version robuste conserve un étalement lisible du
> cœur de la distribution.
"""
)




st.subheader("11. Normalisation : indispensable ou non ?")
n1, n2 = st.columns(2)
with n1:
    st.markdown(
        """
**Exigent une normalisation** *(sensibles aux distances / échelles)* :
- **k-means** et **k-NN** — fondés sur la distance euclidienne : sans normalisation, la
  variable de plus grande amplitude domine (cf. point 9).
- **ACP (PCA)** — maximise la variance ; une variable à forte amplitude capterait
  artificiellement les premiers axes.
- *(aussi : SVM à noyau RBF, régression régularisée Ridge/Lasso, réseaux de neurones.)*
"""
    )
with n2:
    st.markdown(
        """
**N'en ont pas besoin** *(insensibles aux échelles monotones)* :
- **Arbres de décision** (CART) — découpent variable par variable via des **seuils** ;
  une transformation monotone ne change pas les coupures.
- **Forêts aléatoires / Gradient Boosting** — ensembles d'arbres, même propriété.
- *(aussi : Naive Bayes, règles d'association type Apriori sur données catégorielles.)*
"""
    )

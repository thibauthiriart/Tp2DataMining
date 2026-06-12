import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from utils import load_data

st.set_page_config(page_title="Valeurs manquantes", layout="wide")
st.title("Analyse des valeurs manquantes")

try:
    df = load_data()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

st.subheader("Valeurs manquantes par variable")
n_missing = df.isna().sum()
pct_missing = (n_missing / len(df) * 100).round(2)
missing_table = (
    pd.DataFrame({"n_manquantes": n_missing, "pct_manquantes (%)": pct_missing})
    .sort_values("n_manquantes", ascending=False)
)
st.dataframe(missing_table, use_container_width=True)

st.subheader("Heatmap des valeurs manquantes")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.isna(), cbar=False, ax=ax)
ax.set_xlabel("Variables")
ax.set_ylabel("Observations")
st.pyplot(fig)

total_cells = df.size
total_missing = int(df.isna().sum().sum())
pct_global = total_missing / total_cells * 100
st.subheader("Pourcentage global de cellules manquantes")
st.metric("Cellules manquantes", f"{pct_global:.2f} %", f"{total_missing} / {total_cells}")


st.header("Imputation des valeurs manquantes")

vars_avec_nan = [c for c in df.columns if df[c].isna().any() and c.lower() != "id"]
if not vars_avec_nan:
    st.success("Aucune variable ne contient de valeurs manquantes.")
    st.stop()

variable = st.selectbox("Variable à imputer", vars_avec_nan)
est_numerique = pd.api.types.is_numeric_dtype(df[variable])

if est_numerique:
    methodes_dispo = ["suppression", "moyenne", "médiane", "mode", "KNN", "régression linéaire"]
else:
    methodes_dispo = ["suppression", "mode"]
    st.info("Variable qualitative : seules la suppression et le mode sont disponibles.")

methode = st.radio("Méthode d'imputation", methodes_dispo, horizontal=True)

df_impute = df.copy()
n_avant = int(df_impute[variable].isna().sum())
serie_avant = df[variable].copy()

a_coef = b_coef = r2 = None
feature_reg = None

if methode == "suppression":
    df_impute = df_impute.dropna(subset=[variable])

elif methode == "moyenne":
    df_impute[variable] = df_impute[variable].fillna(df_impute[variable].mean())

elif methode == "médiane":
    df_impute[variable] = df_impute[variable].fillna(df_impute[variable].median())

elif methode == "mode":
    df_impute[variable] = df_impute[variable].fillna(df_impute[variable].mode().iloc[0])

elif methode == "KNN":
    k = st.slider("Nombre de voisins k", min_value=1, max_value=10, value=5)
    num_cols = df_impute.select_dtypes(include=np.number).columns.tolist()
    imputer = KNNImputer(n_neighbors=k)
    df_impute[num_cols] = imputer.fit_transform(df_impute[num_cols])

elif methode == "régression linéaire":
    # 7. Régression linéaire simple : une seule variable explicative, paramètres calculés à la main
    num_cols = df_impute.select_dtypes(include=np.number).columns.tolist()
    features_dispo = [c for c in num_cols if c != variable and c.lower() != "id"]
    feature_reg = st.selectbox("Variable explicative", features_dispo)

    paires = df_impute[[feature_reg, variable]].dropna()
    if len(paires) < 2:
        st.warning("Pas assez de couples complets pour estimer la régression.")
    else:
        x = paires[feature_reg].to_numpy()
        y = paires[variable].to_numpy()
        x_bar, y_bar = x.mean(), y.mean()
        sigma_xy = np.mean((x - x_bar) * (y - y_bar))
        sigma_x2 = np.mean((x - x_bar) ** 2)
        a_coef = sigma_xy / sigma_x2
        b_coef = y_bar - a_coef * x_bar
        y_hat = a_coef * x + b_coef
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y_bar) ** 2)
        r2 = 1 - ss_res / ss_tot

        c1, c2, c3 = st.columns(3)
        c1.metric("a (pente)", f"{a_coef:.4f}")
        c2.metric("b (intercept)", f"{b_coef:.4f}")
        c3.metric("R²", f"{r2:.4f}")

        # Imputation : lignes où variable manque mais feature_reg disponible
        masque = df_impute[variable].isna() & df_impute[feature_reg].notna()
        df_impute.loc[masque, variable] = a_coef * df_impute.loc[masque, feature_reg] + b_coef

n_apres = int(df_impute[variable].isna().sum())
c1, c2, c3 = st.columns(3)
c1.metric("NaN avant", n_avant)
c2.metric("NaN après", n_apres)
c3.metric("Lignes après", len(df_impute))

# 9. Comparaison avant / après imputation
if est_numerique:
    st.subheader("Avant vs après imputation")
    serie_apres = df_impute[variable]

    col_avant, col_apres = st.columns(2)
    with col_avant:
        st.markdown("**Avant**")
        st.metric("Moyenne", f"{serie_avant.mean():.2f}")
        st.metric("Médiane", f"{serie_avant.median():.2f}")
        st.metric("Écart-type", f"{serie_avant.std():.2f}")
    with col_apres:
        st.markdown("**Après**")
        st.metric("Moyenne", f"{serie_apres.mean():.2f}")
        st.metric("Médiane", f"{serie_apres.median():.2f}")
        st.metric("Écart-type", f"{serie_apres.std():.2f}")

    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(serie_avant.dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title("Avant")
        st.pyplot(fig)
    with col_hist2:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(serie_apres.dropna(), kde=True, ax=ax, color="seagreen")
        ax.set_title("Après")
        st.pyplot(fig)

st.subheader("Aperçu après imputation")
st.dataframe(df_impute.head())


st.header("Analyse comparée des méthodes")

st.markdown(
    """
**10. Effet sur l'écart-type — moyenne vs KNN**

- L'imputation par la **moyenne** ajoute des valeurs strictement égales à la moyenne :
  ces points ont un écart nul par rapport à la moyenne, donc la somme des carrés des écarts
  augmente moins vite que le nombre d'observations. Conséquence : **l'écart-type diminue**
  artificiellement et la distribution s'aplatit autour du centre.
- L'imputation **KNN** remplace chaque NaN par une combinaison des valeurs des voisins
  les plus proches (sur les autres variables). Les valeurs imputées **conservent de la
  variance** : l'écart-type reste très proche de l'original.

> En pratique, la moyenne sous-estime la dispersion (donc les intervalles de confiance,
> les tests, etc.) alors que KNN préserve la structure de la distribution.

**11. Meilleure variable explicative pour `revenu`**

Comparaison des R² d'une régression linéaire simple sur les paires complètes :
"""
)

if "revenu" in df.columns:
    num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
    candidats = [c for c in num_cols_all if c != "revenu" and c.lower() != "id"]
    scores = []
    for c in candidats:
        paires = df[[c, "revenu"]].dropna()
        if len(paires) < 2:
            continue
        x = paires[c].to_numpy()
        y = paires["revenu"].to_numpy()
        x_bar, y_bar = x.mean(), y.mean()
        var_x = np.mean((x - x_bar) ** 2)
        if var_x == 0:
            continue
        a = np.mean((x - x_bar) * (y - y_bar)) / var_x
        b = y_bar - a * x_bar
        y_hat = a * x + b
        r2_c = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y_bar) ** 2)
        scores.append({"explicative": c, "a": a, "b": b, "R²": r2_c})

    scores_df = pd.DataFrame(scores).sort_values("R²", ascending=False).reset_index(drop=True)
    st.dataframe(scores_df, use_container_width=True)
    if not scores_df.empty:
        meilleure = scores_df.iloc[0]
        st.markdown(
            f"**Meilleure explicative : `{meilleure['explicative']}`** avec un R² de "
            f"**{meilleure['R²']:.3f}**. C'est la variable qui capture la plus grande "
            f"part de la variance de `revenu` ; les autres explicatives ont des R² plus "
            f"faibles, donc une relation linéaire moins informative."
        )

st.markdown(
    """
**12. Méthode recommandée pour ce jeu de données**

- La **suppression** fait perdre des observations : à éviter dès que les NaN sont nombreux
  ou structurés.
- La **moyenne / médiane** déforme la distribution (sous-estimation de l'écart-type, biais
  vers le centre). À réserver aux variables peu manquantes et utilisées sans modélisation
  fine.
- Le **mode** reste indispensable pour les variables qualitatives.
- La **régression linéaire simple** est intéressante quand une variable explicative est
  fortement corrélée à la cible (R² élevé), mais elle injecte des valeurs sur une droite,
  ce qui réduit aussi la dispersion.
- Le **KNN** préserve mieux la variance et tient compte des relations multivariées entre
  variables numériques.

> **Recommandation :** utiliser **KNNImputer** pour les variables quantitatives (avec
> k autour de 5), et le **mode** pour les variables qualitatives. Réserver la régression
> linéaire aux variables ayant un R² élevé avec une explicative bien choisie.
"""
)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import load_data

st.set_page_config(page_title="Distribution & discrétisation", layout="wide")
st.title("Forme de distribution et discrétisation")

try:
    df = load_data()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

quant_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c]) and c.lower() != "id"
]
variable = st.selectbox("Variable quantitative", quant_cols)

serie = df[variable].dropna()
x = serie.to_numpy(dtype=float)

st.subheader("Forme de la distribution")
skew = float(stats.skew(x))
kurt = float(stats.kurtosis(x))  # excès (Fisher) : 0 pour une loi normale

if skew < -0.5:
    interp_skew = "étalée à gauche (asymétrie négative)"
elif skew > 0.5:
    interp_skew = "étalée à droite (asymétrie positive)"
else:
    interp_skew = "approximativement symétrique"

if kurt < -0.5:
    interp_kurt = "platikurtique (queues fines, sommet aplati)"
elif kurt > 0.5:
    interp_kurt = "leptokurtique (queues lourdes, sommet pointu)"
else:
    interp_kurt = "mésokurtique (proche de la loi normale)"

c1, c2 = st.columns(2)
c1.metric("Asymétrie (skew)", f"{skew:.3f}")
c1.caption(f"→ Distribution **{interp_skew}**.")
c2.metric("Aplatissement (kurtosis excédentaire)", f"{kurt:.3f}")
c2.caption(f"→ Distribution **{interp_kurt}**.")

col_hist, col_qq = st.columns(2)

with col_hist:
    st.markdown("**Histogramme + densité**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(x, kde=True, stat="density", ax=ax, color="steelblue")
    ax.set_xlabel(variable)
    st.pyplot(fig)

with col_qq:
    st.markdown("**QQ-plot vs loi normale**")
    fig, ax = plt.subplots(figsize=(5, 4))
    stats.probplot(x, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("steelblue")
    ax.get_lines()[0].set_markeredgecolor("steelblue")
    ax.get_lines()[1].set_color("crimson")
    ax.set_title("")
    st.pyplot(fig)

st.subheader("Test de normalité (Shapiro-Wilk)")
stat_sw, p_value = stats.shapiro(x)
c1, c2 = st.columns(2)
c1.metric("Statistique W", f"{stat_sw:.4f}")
c2.metric("p-value", f"{p_value:.4g}")

if p_value < 0.05:
    st.error(
        f"p-value = {p_value:.4g} < 0.05 → **on rejette** l'hypothèse de normalité : "
        f"`{variable}` ne suit pas une loi normale au seuil de 5 %."
    )
else:
    st.success(
        f"p-value = {p_value:.4g} ≥ 0.05 → **on ne rejette pas** l'hypothèse de normalité : "
        f"`{variable}` est compatible avec une loi normale au seuil de 5 %."
    )

st.divider()

st.subheader("Discrétisation")
col_k, col_m = st.columns(2)
k = col_k.slider("Nombre de classes k", min_value=2, max_value=10, value=5)
methode = col_m.radio(
    "Méthode",
    ["largeur égale (pd.cut)", "fréquence égale (pd.qcut)"],
    horizontal=True,
)

if methode.startswith("largeur"):
    classes = pd.cut(serie, bins=k)
else:
    # qcut peut échouer si trop de doublons aux bornes -> duplicates='drop'
    classes = pd.qcut(serie, q=k, duplicates="drop")
    n_eff = classes.cat.categories.size
    if n_eff < k:
        st.info(
            f"Seulement {n_eff} classes obtenues (au lieu de {k}) : valeurs trop "
            f"concentrées sur certaines bornes, classes vides fusionnées."
        )

effectifs = (
    classes.value_counts(sort=False)
    .rename_axis("classe")
    .reset_index(name="effectif")
)
effectifs["classe"] = effectifs["classe"].astype(str)

col_g, col_t = st.columns([2, 1])
with col_g:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=effectifs, x="classe", y="effectif", ax=ax, color="steelblue")
    ax.set_xlabel(f"Classes de {variable}")
    ax.set_ylabel("Effectif")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
with col_t:
    st.markdown("**Effectifs par classe**")
    st.dataframe(effectifs, use_container_width=True, hide_index=True)

st.divider()

st.subheader("10. Quelles variables suivent approximativement une loi normale ?")
rows = []
for c in quant_cols:
    xc = df[c].dropna().to_numpy(dtype=float)
    if len(xc) < 3:
        continue
    p = stats.shapiro(xc).pvalue
    rows.append(
        {
            "variable": c,
            "skew": round(float(stats.skew(xc)), 3),
            "kurtosis": round(float(stats.kurtosis(xc)), 3),
            "p-value Shapiro": round(float(p), 4),
            "normale (5%)": "oui" if p >= 0.05 else "non",
        }
    )
normal_df = pd.DataFrame(rows)
st.dataframe(normal_df, use_container_width=True, hide_index=True)

normales = normal_df.loc[normal_df["normale (5%)"] == "oui", "variable"].tolist()
st.markdown(
    f"""
Au seuil de 5 % (Shapiro-Wilk), les variables **non rejetées** comme normales sont :
{", ".join(f"`{v}`" for v in normales) if normales else "*aucune*"}.

Ce sont aussi celles dont l'asymétrie et le kurtosis excédentaire restent proches de 0.
Les variables de type **montant** (`revenu`, `solde_compte`, `depenses_mensuelles`) sont
généralement étalées à droite et donc rejetées.
"""
)

st.subheader("11. Effet d'un outlier sur la discrétisation à largeur égale")
cible = "revenu" if "revenu" in quant_cols else variable
base = df[cible].dropna()
extreme = base.max() * 3  # outlier artificiel
avec_out = pd.concat([base, pd.Series([extreme])], ignore_index=True)

cut_sans = pd.cut(base, bins=k).value_counts(sort=False)
cut_avec = pd.cut(avec_out, bins=k).value_counts(sort=False)

comp = pd.DataFrame(
    {
        "classe (sans outlier)": [str(i) for i in cut_sans.index],
        "effectif (sans)": cut_sans.values,
    }
)
comp_avec = pd.DataFrame(
    {
        "classe (avec outlier)": [str(i) for i in cut_avec.index],
        "effectif (avec)": cut_avec.values,
    }
)
cc1, cc2 = st.columns(2)
cc1.markdown(f"**`{cible}` sans outlier**")
cc1.dataframe(comp, use_container_width=True, hide_index=True)
cc2.markdown(f"**`{cible}` avec un outlier ({extreme:.0f})**")
cc2.dataframe(comp_avec, use_container_width=True, hide_index=True)

st.markdown(
    """
Un seul outlier **étire le domaine** `[min ; max]` utilisé par `pd.cut`. Comme les classes
sont de **largeur égale**, elles deviennent toutes beaucoup plus larges : la quasi-totalité
des observations se retrouve **tassée dans la première classe**, et les classes supérieures
sont **vides ou presque** (seul l'outlier y figure).

> `pd.cut` (largeur égale) est donc **très sensible aux valeurs extrêmes** : la
> discrétisation perd son pouvoir discriminant.
"""
)

# --- 12. Quelle méthode pour équilibrer les effectifs ? -----------------------
st.subheader("12. Méthode pour équilibrer les effectifs par classe")
st.markdown(
    """
Pour obtenir des classes **d'effectifs équilibrés**, il faut utiliser la discrétisation à
**fréquence égale** (`pd.qcut`), basée sur les **quantiles** : chaque classe contient
environ le même nombre d'observations (≈ n/k).

- `pd.cut` (largeur égale) découpe l'**axe des valeurs** en intervalles identiques → les
  effectifs dépendent de la densité et sont déséquilibrés si la distribution est asymétrique.
- `pd.qcut` (fréquence égale) découpe selon les **rangs** → effectifs homogènes, et c'est
  aussi **robuste aux outliers** (un point extrême ne fait que peupler la dernière classe).

> **Choix : `pd.qcut`** lorsque l'objectif est d'équilibrer les effectifs par classe.
"""
)

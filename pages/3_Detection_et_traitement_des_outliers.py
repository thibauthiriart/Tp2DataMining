import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

st.set_page_config(page_title="Outliers", layout="wide")
st.title("Détection et traitement des outliers")

try:
    df = load_data()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()


def skewness(x: np.ndarray) -> float:
    """Coefficient d'asymétrie de Fisher-Pearson (g1), calculé à la main."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


# --- 1. Sélection d'une variable quantitative ---------------------------------
quant_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c]) and c.lower() != "id"
]
variable = st.selectbox("Variable quantitative", quant_cols)

# Série de travail : on garde les id pour pouvoir lister les outliers
id_col = next((c for c in df.columns if c.lower() == "id"), None)
work = df[[variable]].copy()
work["__id__"] = df[id_col] if id_col is not None else df.index
work = work.dropna(subset=[variable])

serie = work[variable]
ids = work["__id__"]
x = serie.to_numpy(dtype=float)


# --- Méthodes de détection ----------------------------------------------------
def detect_iqr(values: np.ndarray, k: float):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    mask = (values < low) | (values > high)
    info = {"Q1": q1, "Q3": q3, "IQR": iqr, "borne basse": low, "borne haute": high}
    return mask, low, high, info


def detect_zscore(values: np.ndarray, s: float):
    mean = values.mean()
    std = values.std(ddof=0)
    z = (values - mean) / std if std != 0 else np.zeros_like(values)
    mask = np.abs(z) > s
    # bornes équivalentes pour la winsorization
    low, high = mean - s * std, mean + s * std
    info = {"moyenne": mean, "écart-type": std, "seuil |z|": s,
            "borne basse": low, "borne haute": high}
    return mask, low, high, info


def detect_mad(values: np.ndarray, thr: float):
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        z_mod = np.zeros_like(values)
    else:
        z_mod = 0.6745 * (values - med) / mad
    mask = np.abs(z_mod) > thr
    # bornes équivalentes : |0.6745 (x - med)/MAD| = thr
    delta = thr * mad / 0.6745 if mad != 0 else 0.0
    low, high = med - delta, med + delta
    info = {"médiane": med, "MAD": mad, "seuil |z_mod|": thr,
            "borne basse": low, "borne haute": high}
    return mask, low, high, info


# --- 5. Choix de la méthode via des onglets -----------------------------------
st.subheader("Méthode de détection")
tab_iqr, tab_z, tab_mad = st.tabs(["IQR (Tukey)", "Z-score", "Z-score modifié (MAD)"])

with tab_iqr:
    k = st.slider("k (IQR)", 0.5, 5.0, 1.5, 0.1)
    mask_iqr, low_iqr, high_iqr, info_iqr = detect_iqr(x, k)
    st.write({kk: round(vv, 3) for kk, vv in info_iqr.items()})
    st.metric("Outliers détectés", int(mask_iqr.sum()))
    st.write("**id des outliers :**", sorted(ids[mask_iqr].tolist()))

with tab_z:
    s = st.slider("seuil s (|z| >)", 1.0, 5.0, 2.0, 0.1)
    mask_z, low_z, high_z, info_z = detect_zscore(x, s)
    st.write({kk: round(vv, 3) for kk, vv in info_z.items()})
    st.metric("Outliers détectés", int(mask_z.sum()))
    st.write("**id des outliers :**", sorted(ids[mask_z].tolist()))

with tab_mad:
    thr = st.slider("seuil |z_mod| >", 2.0, 6.0, 3.5, 0.1)
    mask_mad, low_mad, high_mad, info_mad = detect_mad(x, thr)
    st.write({kk: round(vv, 3) for kk, vv in info_mad.items()})
    st.metric("Outliers détectés", int(mask_mad.sum()))
    st.write("**id des outliers :**", sorted(ids[mask_mad].tolist()))

methodes = {
    "IQR (Tukey)": (mask_iqr, low_iqr, high_iqr),
    "Z-score": (mask_z, low_z, high_z),
    "Z-score modifié (MAD)": (mask_mad, low_mad, high_mad),
}

st.divider()
choix = st.radio(
    "Méthode retenue pour les visualisations et le traitement",
    list(methodes.keys()),
    horizontal=True,
)
mask, low, high = methodes[choix]
mask = np.asarray(mask)

# --- 6. Visualisations : boxplot + histogramme avec outliers colorés ----------
st.subheader("Visualisations")
col_box, col_hist = st.columns(2)

with col_box:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(y=serie, ax=ax, color="skyblue", fliersize=0)
    # On superpose les points : outliers en rouge, le reste en gris
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(x)) - 0.5) * 0.1
    ax.scatter(jitter[~mask], x[~mask], color="gray", alpha=0.5, s=15, label="normal")
    ax.scatter(jitter[mask], x[mask], color="crimson", s=30, label="outlier")
    ax.set_ylabel(variable)
    ax.legend()
    ax.set_title("Boxplot")
    st.pyplot(fig)

with col_hist:
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.histogram_bin_edges(x, bins=30)
    ax.hist(x[~mask], bins=bins, color="steelblue", label="normal")
    ax.hist(x[mask], bins=bins, color="crimson", label="outlier")
    for b in (low, high):
        ax.axvline(b, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(variable)
    ax.set_ylabel("Fréquence")
    ax.legend()
    ax.set_title("Histogramme")
    st.pyplot(fig)

# --- 7. Stratégie de traitement -----------------------------------------------
st.subheader("Traitement")
strategie = st.radio(
    "Stratégie",
    ["conserver", "supprimer", "winsorization", "transformation log"],
    horizontal=True,
)

x_trait = x.copy()
note = ""
if strategie == "conserver":
    note = "Aucune modification : les outliers sont conservés."
elif strategie == "supprimer":
    x_trait = x[~mask]
    note = f"{int(mask.sum())} observation(s) supprimée(s)."
elif strategie == "winsorization":
    x_trait = np.clip(x, low, high)
    note = "Les valeurs hors bornes sont ramenées aux bornes [borne basse ; borne haute]."
elif strategie == "transformation log":
    if np.any(x <= 0):
        shift = -x.min() + 1.0
        x_trait = np.log(x + shift)
        note = f"Présence de valeurs ≤ 0 : application de log(x + {shift:.2f})."
    else:
        x_trait = np.log(x)
        note = "Application de log(x)."
st.caption(note)

# --- 8. Statistiques avant / après --------------------------------------------
st.subheader("Statistiques avant / après traitement")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Avant**")
    st.metric("Moyenne", f"{x.mean():.2f}")
    st.metric("Médiane", f"{np.median(x):.2f}")
    st.metric("Écart-type", f"{x.std(ddof=0):.2f}")
with col_b:
    st.markdown("**Après**")
    st.metric("Moyenne", f"{x_trait.mean():.2f}")
    st.metric("Médiane", f"{np.median(x_trait):.2f}")
    st.metric("Écart-type", f"{x_trait.std(ddof=0):.2f}")

st.divider()

# --- 9. Comparaison des trois méthodes sur `revenu` ---------------------------
st.subheader("9. Comparaison des trois méthodes")
cible = "revenu" if "revenu" in quant_cols else variable
xc = df[cible].dropna().to_numpy(dtype=float)
m_iqr, *_ = detect_iqr(xc, 1.5)
m_z, *_ = detect_zscore(xc, 2.0)
m_mad, *_ = detect_mad(xc, 3.5)
comp = pd.DataFrame(
    {
        "méthode": ["IQR (k=1.5)", "Z-score (s=2)", "Z-score modifié (3.5)"],
        "n_outliers": [int(m_iqr.sum()), int(m_z.sum()), int(m_mad.sum())],
    }
)
st.markdown(f"Variable analysée : **`{cible}`**")
st.dataframe(comp, use_container_width=True, hide_index=True)

# --- 10. Pourquoi le z-score sous-détecte ------------------------------------
st.subheader("10. Pourquoi le z-score classique sous-détecte les outliers nombreux ?")
st.markdown(
    """
Le z-score repose sur la **moyenne** $\\bar{x}$ et l'**écart-type** $\\sigma$,
deux statistiques **non robustes** : elles sont elles-mêmes calculées à partir des
données *outliers compris*.

- Quand les valeurs extrêmes sont **nombreuses** (ou très grandes), elles **tirent la
  moyenne** vers elles et surtout **gonflent l'écart-type** $\\sigma$.
- Le dénominateur $\\sigma$ augmentant, les z-scores $z_i = (x_i-\\bar{x})/\\sigma$
  **diminuent** mécaniquement : des points pourtant aberrants passent sous le seuil.
- C'est l'effet de **masquage** (*masking*) : les outliers se cachent mutuellement.

> À l'inverse, la **médiane** et le **MAD** (z-score modifié) sont robustes : ils ne sont
> presque pas affectés par une fraction d'observations extrêmes, d'où une détection plus
> fiable quand les outliers sont nombreux.
"""
)

# --- 11. Transformation log sur depenses_mensuelles --------------------------
st.subheader("11. Transformation log sur `depenses_mensuelles`")
col_dep = next((c for c in df.columns if c.lower() == "depenses_mensuelles"), None)
if col_dep is None:
    st.info("La variable `depenses_mensuelles` est absente du jeu de données.")
else:
    d = df[col_dep].dropna().to_numpy(dtype=float)
    if np.any(d <= 0):
        shift = -d.min() + 1.0
        d_log = np.log(d + shift)
        transfo = f"log(x + {shift:.2f})"
    else:
        d_log = np.log(d)
        transfo = "log(x)"
    sk_avant = skewness(d)
    sk_apres = skewness(d_log)

    c1, c2 = st.columns(2)
    c1.metric("Asymétrie avant", f"{sk_avant:.3f}")
    c2.metric("Asymétrie après", f"{sk_apres:.3f}", f"{sk_apres - sk_avant:+.3f}")

    fig, (axg, axd) = plt.subplots(1, 2, figsize=(10, 3.5))
    sns.histplot(d, kde=True, ax=axg, color="steelblue")
    axg.set_title(f"Avant (skew={sk_avant:.2f})")
    axg.set_xlabel(col_dep)
    sns.histplot(d_log, kde=True, ax=axd, color="seagreen")
    axd.set_title(f"Après {transfo} (skew={sk_apres:.2f})")
    axd.set_xlabel(f"{transfo}")
    st.pyplot(fig)

    pertinente = abs(sk_apres) < abs(sk_avant)
    verdict = "**pertinente**" if pertinente else "**peu utile**"
    explication = (
        "En rapprochant l'asymétrie de 0, le log **comprime les grandes valeurs** et "
        "**symétrise** la distribution : elle devient plus proche d'une loi normale, ce "
        "qui stabilise la variance et atténue l'influence des valeurs extrêmes."
        if pertinente
        else "L'asymétrie ne se réduit pas : la transformation log n'est pas justifiée ici."
    )
    st.markdown(
        f"""
La transformation log est ici {verdict}.

- Asymétrie **avant** : `{sk_avant:.3f}` — une valeur nettement positive signale une
  distribution **étalée vers la droite** (longue queue de fortes dépenses), typique des
  variables de type montant/revenu.
- Asymétrie **après** `{transfo}` : `{sk_apres:.3f}`.
- {explication}

> Une transformation log est pertinente précisément quand elle **réduit l'asymétrie
> positive** d'une variable strictement positive et fortement étalée à droite.
"""
    )

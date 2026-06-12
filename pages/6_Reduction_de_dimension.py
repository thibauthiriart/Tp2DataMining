import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from utils import prepare_clients, load_iris_df

st.set_page_config(page_title="Réduction de dimension", layout="wide")
st.title("Réduction de dimension : ACP, MDS, Isomap")

dataset = st.selectbox("Jeu de données", ["Iris", "Clients banque"])

if dataset == "Iris":
    df = load_iris_df()
    label_col = "species"
    feat_cols = [c for c in df.columns if c != label_col]
    labels = df[label_col]
else:
    df = prepare_clients()
    id_col = next((c for c in df.columns if c.lower() == "id"), None)
    feat_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != id_col
    ]
    label_col = None
    labels = None

X = df[feat_cols].to_numpy(dtype=float)

st.caption(
    f"**{dataset}** — {X.shape[0]} individus, {X.shape[1]} variables quantitatives "
    f"(déjà standardisées)."
    + (" Coloration par espèce disponible." if labels is not None else "")
)


def scatter_2d(ax, coords, title):
    if labels is not None:
        for lab in pd.unique(labels):
            m = labels.to_numpy() == lab
            ax.scatter(coords[m, 0], coords[m, 1], label=str(lab), alpha=0.75, s=25)
        ax.legend(fontsize=8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=22, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")


st.divider()

st.header("2. Matrice de corrélation des variables quantitatives")
corr = df[feat_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f",
            square=True, linewidths=0.5, ax=ax)
st.pyplot(fig)

mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
pairs = corr.where(mask_upper).stack()
pairs_sorted = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
top = pairs_sorted.head(3)

lignes = [
    f"- `{a}` ↔ `{b}` : **r = {r:+.2f}** "
    f"({'forte' if abs(r) >= 0.7 else 'modérée' if abs(r) >= 0.4 else 'faible'} "
    f"corrélation {'positive' if r > 0 else 'négative'})"
    for (a, b), r in top.items()
]
pa, pb = top.index[0]
st.markdown(
    "**Commentaire automatique — paires les plus corrélées :**\n"
    + "\n".join(lignes)
    + (
        f"\n\nLa paire la plus liée est **`{pa}` / `{pb}`** (r = {top.iloc[0]:+.2f}). "
        "Des variables fortement corrélées sont **redondantes** : elles portent une "
        "information commune, ce qui justifie une **réduction de dimension** — l'ACP va "
        "précisément regrouper cette information sur un même axe."
    )
)

st.divider()

st.header("3. ACP — variance expliquée et diagramme des éboulis")
pca = PCA()
scores = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

var_table = pd.DataFrame(
    {
        "Composante": [f"PC{i + 1}" for i in range(len(evr))],
        "Variance expliquée (%)": (evr * 100).round(2),
        "Variance cumulée (%)": (cum * 100).round(2),
    }
)
col_t, col_p = st.columns([1, 1.3])
with col_t:
    st.dataframe(var_table, use_container_width=True, hide_index=True)
with col_p:
    comps = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(comps, evr * 100, color="steelblue", alpha=0.8, label="variance expliquée")
    ax.plot(comps, cum * 100, color="crimson", marker="o", label="variance cumulée")
    ax.axhline(90, color="gray", ls="--", lw=1)
    ax.text(comps[-1], 91, "90 %", color="gray", ha="right", fontsize=8)
    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Variance (%)")
    ax.set_xticks(comps)
    ax.set_title("Scree plot")
    ax.legend()
    st.pyplot(fig)

n90 = int(np.argmax(cum >= 0.90) + 1)
st.metric(
    "4. Composantes nécessaires pour expliquer ≥ 90 % de la variance",
    f"{n90} / {len(evr)}",
    f"{cum[n90 - 1] * 100:.1f} % cumulés",
)
st.caption(
    f"Les **{n90}** premières composantes suffisent à résumer 90 % de l'information : "
    f"on pourrait passer de {len(evr)} à {n90} dimensions avec une perte minime."
)

st.divider()

st.header("5. Cercle des corrélations (variables vs PC1/PC2)")
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

col_c, col_txt = st.columns([1, 1])
with col_c:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle((0, 0), 1, color="gray", fill=False, lw=1.2))
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    for i, name in enumerate(feat_cols):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 head_width=0.03, head_length=0.04, color="steelblue",
                 length_includes_head=True)
        ax.text(loadings[i, 0] * 1.12, loadings[i, 1] * 1.12, name,
                color="darkred", ha="center", va="center", fontsize=8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f} %)")
    ax.set_title("Cercle des corrélations")
    st.pyplot(fig)
with col_txt:
    st.markdown(
        """
**Lecture du cercle :**
- Une flèche proche du cercle (longue) est bien représentée par le plan PC1–PC2.
- Deux flèches dans la même direction désignent des variables corrélées positivement ;
  en directions opposées, corrélées négativement ; **orthogonales (≈ 90°), peu corrélées.
- La projection d'une flèche sur un axe donne sa corrélation avec cette composante. 
"""
    )

st.divider()

st.header("6. Projection des individus sur les deux premières composantes")
fig, ax = plt.subplots(figsize=(7, 5))
scatter_2d(ax, scores[:, :2], "Projection ACP (PC1 vs PC2)")
ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f} %)")
ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f} %)")
st.pyplot(fig)

st.divider()

st.header("7. MDS (Multi-Dimensional Scaling)")
mds_choix = st.selectbox(
    "Type de MDS",
    ["Métrique (metric=True)", "Non métrique (metric=False)"],
)
is_metric = mds_choix.startswith("Métrique")
mds = MDS(
    n_components=2,
    dissimilarity="euclidean",
    random_state=42,
    metric=is_metric,
    normalized_stress=True,
)
coords_mds = mds.fit_transform(X)

col_m, col_ms = st.columns([1.2, 1])
with col_m:
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_2d(ax, coords_mds, f"MDS {'métrique' if is_metric else 'non métrique'}")
    st.pyplot(fig)
with col_ms:
    st.metric("Stress (normalisé, Kruskal stress-1)", f"{mds.stress_:.4f}")
    st.markdown(
        """
**À quoi correspond le stress ?**

Le **stress** mesure l'écart entre les distances **dans l'espace d'origine** et les
distances **dans la projection 2D** :

$$\\text{stress} = \\sqrt{\\dfrac{\\sum_{i<j}\\,(d_{ij} - \\hat{d}_{ij})^2}{\\sum_{i<j}\\, d_{ij}^2}}$$

**Son sens :** plus il est **proche de 0**, mieux la projection **préserve les distances**
(donc la structure des données). Repères usuels de Kruskal : `< 0.05` excellent,
`0.05–0.10` bon, `0.10–0.20` acceptable, `> 0.20` médiocre.
"""
    )

st.divider()

st.header("8. Isomap")
k = st.slider("n_neighbors (k)", min_value=3, max_value=30, value=5)
iso = Isomap(n_components=2, n_neighbors=k)
coords_iso = iso.fit_transform(X)

col_i, col_ie = st.columns([1.2, 1])
with col_i:
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_2d(ax, coords_iso, f"Isomap (k = {k})")
    st.pyplot(fig)
with col_ie:
    st.metric("reconstruction_error()", f"{iso.reconstruction_error():.4f}")
    st.caption(
        "L'erreur de reconstruction mesure la part de la structure des **distances "
        "géodésiques** (le long de la variété) non restituée par la projection 2D : "
        "plus elle est faible, meilleure est la préservation."
    )

st.divider()

st.header("9. ACP, MDS et Isomap côte à côte")
projections = [
    ("ACP", scores[:, :2]),
    (f"MDS ({'métrique' if is_metric else 'non métrique'})", coords_mds),
    (f"Isomap (k = {k})", coords_iso),
]
cols = st.columns(3)
for col, (nom, coords) in zip(cols, projections):
    with col:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        scatter_2d(ax, coords, nom)
        st.pyplot(fig)

st.divider()

st.header("10. Comparaison des trois méthodes")
if dataset == "Iris":
    st.markdown(
        """
Sur **Iris**, les trois projections séparent nettement **`setosa`** (très distinct) des
deux autres espèces. La frontière entre **`versicolor`** et **`virginica`** est plus floue
car ces deux classes se chevauchent réellement.

- **ACP** : sépare bien les classes car l'essentiel de la variance d'Iris est porté par les
  dimensions des pétales, qui sont précisément discriminantes — une méthode **linéaire**
  suffit ici.
- **MDS (euclidien)** : projection très proche de l'ACP (mêmes distances euclidiennes
  préservées), bonne séparation.
- **Isomap** : préserve les distances **géodésiques** et resserre souvent davantage les
  classes le long de la variété ; la séparation `setosa` reste franche.

> Iris étant un jeu **quasi linéairement séparable**, **ACP/MDS** préservent déjà très bien
> la structure des classes ; Isomap n'apporte ici qu'un gain marginal.
"""
    )
else:
    st.markdown(
        """
Sur **Clients banque** (non étiqueté), il faut chercher des **groupes naturels** :

- L'**ACP** étale les individus selon les axes de plus forte variance (typiquement `revenu`,
  `solde_compte`, `depenses_mensuelles`) : on repère parfois un dégradé plus qu'un découpage
  net en clusters.
- **MDS** donne une image proche de l'ACP (distances euclidiennes).
- **Isomap** peut révéler une structure **courbe** invisible à l'ACP si les clients se
  répartissent le long d'une variété non linéaire.

> Observez si une des projections fait apparaître des **paquets séparés** : c'est un indice
> qu'un *clustering* (k-means, CAH) trouvera des segments. Si le nuage reste **continu**, les
> clients forment plutôt un **continuum** sans frontières marquées.
"""
    )

st.header("11. Méthode linéaire (ACP) vs non linéaire (Isomap)")
st.markdown(
    """
**Différence fondamentale :**

- L'**ACP** est **linéaire** : elle cherche des **combinaisons linéaires** des variables
  (des axes droits) maximisant la variance. Elle ne « voit » que les distances **euclidiennes
  globales** et ne peut pas dérouler une structure courbe.
- **Isomap** est **non linéaire** : il construit un **graphe de voisinage** (`n_neighbors`)
  puis préserve les **distances géodésiques** (le plus court chemin *le long de la variété*).
  Il peut **« déplier »** des structures incurvées (le cas d'école : le *Swiss roll*).

**Quand préférer chacune ?**

| Préférer **ACP** | Préférer **Isomap** |
|---|---|
| Données ~ linéairement structurées | Variété **courbe / repliée** |
| Besoin d'**interprétabilité** (loadings, cercle des corrélations) | Structure **non linéaire** à révéler |
| Rapidité, robustesse, projection de nouveaux points | Voisinages **locaux** porteurs de sens |

> En résumé : **ACP** = projection linéaire interprétable et rapide ; **Isomap** = capture des
> géométries non linéaires que l'ACP aplatirait à tort.
"""
)

st.header("12. Influence de `n_neighbors` dans Isomap (k = 3 vs k = 30)")
iso_lo = Isomap(n_components=2, n_neighbors=3).fit(X)
coords_lo = iso_lo.transform(X)
iso_hi = Isomap(n_components=2, n_neighbors=30).fit(X)
coords_hi = iso_hi.transform(X)

c_lo, c_hi = st.columns(2)
with c_lo:
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter_2d(ax, coords_lo, "Isomap k = 3")
    st.pyplot(fig)
    st.metric("reconstruction_error (k=3)", f"{iso_lo.reconstruction_error():.4f}")
with c_hi:
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter_2d(ax, coords_hi, "Isomap k = 30")
    st.pyplot(fig)
    st.metric("reconstruction_error (k=30)", f"{iso_hi.reconstruction_error():.4f}")

st.markdown(
    """
- **k petit (= 3)** : graphe de voisinage **très local**, sensible au **bruit** ; il peut se
  **fragmenter** (voisinages déconnectés) et produire des projections morcelées ou distordues.
- **k grand (= 30)** : le graphe relie des points **éloignés** par des « raccourcis », Isomap
  **se rapproche d'une ACP** (distances de plus en plus euclidiennes) et **lisse** la structure
  fine, perdant le caractère non linéaire.

> `n_neighbors` arbitre entre **fidélité locale** (k petit) et **stabilité globale** (k grand) :
> trop petit, on capte le bruit ; trop grand, on perd l'intérêt de la méthode. Une valeur
> **intermédiaire** est généralement le meilleur compromis.
"""
)

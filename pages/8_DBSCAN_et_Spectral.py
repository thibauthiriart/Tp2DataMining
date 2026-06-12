import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, SpectralClustering
from utils import prepare_clients, load_iris_df, load_data

st.set_page_config(page_title="DBSCAN & Spectral", layout="wide")
st.title("DBSCAN et Spectral Clustering")


dataset = st.selectbox("Jeu de données", ["Iris", "Clients banque"])

if dataset == "Iris":
    df = load_iris_df()
    feat_cols = [c for c in df.columns if c != "species"]
    ids = None
else:
    df = prepare_clients()
    id_col = next((c for c in df.columns if c.lower() == "id"), None)
    feat_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != id_col
    ]
    ids = df[id_col]

X = df[feat_cols].to_numpy(dtype=float)

pca_plane = PCA(n_components=2).fit(X)
acp = pca_plane.transform(X)


def plot_clusters(coords, labels, title, noise_label=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.asarray(labels)
    if noise_label is not None:
        mask_noise = labels == noise_label
        ax.scatter(coords[mask_noise, 0], coords[mask_noise, 1],
                   c="black", marker="x", s=35, label="bruit (-1)")
        normal = ~mask_noise
        ax.scatter(coords[normal, 0], coords[normal, 1],
                   c=labels[normal], cmap="tab10", alpha=0.8, s=25)
        if mask_noise.any():
            ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.8, s=25)
    ax.set_xlabel(f"PC1 ({pca_plane.explained_variance_ratio_[0] * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca_plane.explained_variance_ratio_[1] * 100:.1f} %)")
    ax.set_title(title)
    return fig


st.divider()

st.header("2. DBSCAN")

st.markdown(
    """
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) repose sur deux
paramètres clés :

- eps : le rayon du voisinage. Deux points sont voisins si leur distance est inférieure à eps.
- min_samples : le nombre minimum de points (lui-même compris) devant se trouver dans le
  rayon eps pour qu'un point soit considéré comme un point cœur et amorce un cluster dense.

Un point qui n'est ni cœur ni atteignable depuis un point cœur est étiqueté bruit (label -1).
"""
)

c1, c2 = st.columns(2)
with c1:
    min_samples = st.slider("min_samples", min_value=2, max_value=20, value=5)
with c2:
    pass

nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
dist, _ = nn.kneighbors(X)
kdist = np.sort(dist[:, -1])
eps_defaut = float(round(np.percentile(kdist, 90), 2))
max_kdist = float(round(kdist.max(), 2))

with c2:
    eps = st.slider(
        "eps (rayon)",
        min_value=0.1, max_value=max(max_kdist, 0.2),
        value=min(eps_defaut, max_kdist), step=0.05,
    )

col_k, col_kt = st.columns([1.3, 1])
with col_k:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(kdist)), kdist, color="steelblue")
    ax.axhline(eps, color="red", ls="--", lw=1.2, label=f"eps = {eps}")
    ax.set_xlabel("Points triés par distance croissante")
    ax.set_ylabel(f"Distance au {min_samples}-ième plus proche voisin")
    ax.set_title("Courbe des k-distances")
    ax.legend()
    st.pyplot(fig)
with col_kt:
    st.markdown(
        """
Lecture : la courbe monte lentement puis décolle brutalement. Le coude (changement de pente)
sépare les points denses des points isolés : sa hauteur donne un bon eps. La valeur par défaut
proposée ici est le 90e percentile des k-distances ; ajustez le slider eps pour le placer sur
le coude.
"""
    )

db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = int((db_labels == -1).sum())

m1, m2, m3 = st.columns(3)
m1.metric("Clusters trouvés", n_clusters)
m2.metric("Points de bruit (-1)", n_noise)
m3.metric("Points classés", len(db_labels) - n_noise)

fig = plot_clusters(acp, db_labels, f"DBSCAN (eps={eps}, min_samples={min_samples})",
                    noise_label=-1)
st.pyplot(fig)

st.divider()

st.header("3. Spectral Clustering")

st.markdown(
    """
Principe du Spectral Clustering :

1. Construire un graphe de similarité entre les points (chaque point est un nœud, les arêtes
   pondèrent la proximité — via un noyau RBF ou les plus proches voisins).
2. Former la matrice d'affinité puis le Laplacien du graphe.
3. Calculer les premiers vecteurs propres du Laplacien (plongement spectral), puis appliquer
   un clustering simple (K-Means) sur ces vecteurs propres.

L'idée : couper le graphe en composantes faiblement connectées entre elles, ce qui permet de
séparer des formes non convexes que K-Means ne sait pas isoler.
"""
)

s1, s2 = st.columns(2)
with s1:
    k_spec = st.slider("Nombre de clusters k", min_value=2, max_value=10, value=3)
with s2:
    affinity = st.radio("Affinité", ["rbf", "nearest_neighbors"], horizontal=True)

spec = SpectralClustering(
    n_clusters=k_spec, affinity=affinity, random_state=42
)
spec_labels = spec.fit_predict(X)

fig = plot_clusters(acp, spec_labels,
                    f"Spectral Clustering (k={k_spec}, affinity={affinity})")
st.pyplot(fig)

st.divider()

st.header("4. Pour quel type de structure ?")
st.markdown(
    """
DBSCAN est particulièrement adapté aux clusters de forme arbitraire (non convexes, allongés,
imbriqués) et de densité comparable, séparés par des zones de faible densité. Il identifie en
plus le bruit, sans qu'on lui impose le nombre de clusters.

Spectral Clustering est adapté aux structures non convexes définies par la connectivité du
graphe (cercles concentriques, spirales, deux lunes) : il sépare selon les liens de voisinage
plutôt que selon la distance à un centre.

Sur Iris : les classes sont globalement convexes et compactes (setosa bien séparée, versicolor
et virginica qui se chevauchent). K-Means y est déjà très efficace. DBSCAN et Spectral
n'apportent pas d'avantage décisif : DBSCAN tend à fusionner versicolor et virginica en un seul
amas dense (ou à les marquer en bruit selon eps), et Spectral retrouve à peu près la même
partition que K-Means. L'intérêt de ces méthodes apparaît surtout sur des structures non
convexes, que Iris ne présente pas.
"""
)

st.header("5. Avantage majeur de DBSCAN")
st.markdown(
    """
DBSCAN n'exige pas de fixer le nombre de clusters à l'avance (contrairement à K-Means, à la CAH
coupée à un seuil et au Spectral Clustering) : il le déduit de la densité des données. Il
détecte aussi des clusters de forme arbitraire et isole nativement le bruit / les outliers,
au lieu de forcer chaque point dans un cluster.
"""
)

st.header("6. Inconvénient majeur de DBSCAN")
st.markdown(
    """
DBSCAN utilise un eps et un min_samples uniques pour tout le jeu de données : il suppose donc
une densité homogène. Quand les clusters ont des densités très différentes, aucun couple
(eps, min_samples) ne convient à tous : un eps adapté aux clusters denses fragmente les clusters
peu denses (leurs points deviennent du bruit), tandis qu'un eps adapté aux clusters peu denses
fusionne les clusters denses voisins. Cette incapacité à gérer des densités hétérogènes est sa
principale limite (que des variantes comme HDBSCAN cherchent à corriger).
"""
)

st.divider()

st.header("7. DBSCAN et détection d'outliers (Clients banque)")
if dataset != "Clients banque":
    st.info(
        "Cette analyse compare le bruit DBSCAN aux outliers du TP 1 et n'a de sens que "
        "pour Clients banque (jeu du TP 1). Sélectionnez ce jeu de données ci-dessus."
    )
else:
    noise_ids = sorted(int(i) for i in ids[db_labels == -1])
    st.markdown(
        f"Avec les paramètres courants (eps = {eps}, min_samples = {min_samples}), "
        f"DBSCAN classe {len(noise_ids)} client(s) comme bruit, c'est-à-dire comme des "
        f"points isolés ne participant à aucune zone dense — autrement dit des outliers "
        f"multivariés."
    )
    st.write("id des clients détectés comme bruit par DBSCAN :", noise_ids)

    raw = load_data()
    raw_id = next((c for c in raw.columns if c.lower() == "id"), None)
    tp1_sets = {}
    for col in ("revenu", "solde_compte"):
        if col in raw.columns:
            x = raw[col]
            q1, q3 = np.nanpercentile(x, [25, 75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (x < low) | (x > high)
            tp1_sets[col] = set(int(i) for i in raw.loc[mask, raw_id])

    tp1_union = sorted(set().union(*tp1_sets.values())) if tp1_sets else []
    st.markdown(
        "Outliers détectés en TP 1 (méthode IQR de Tukey, k = 1.5, sur les variables "
        "winsorisées `revenu` et `solde_compte`, données brutes) :"
    )
    for col, s in tp1_sets.items():
        st.write(f"- `{col}` : {sorted(s)}")

    inter = sorted(set(noise_ids) & set(tp1_union))
    only_db = sorted(set(noise_ids) - set(tp1_union))
    only_tp1 = sorted(set(tp1_union) - set(noise_ids))

    cmp = pd.DataFrame(
        {
            "ensemble": ["Bruit DBSCAN", "Outliers TP 1 (union)",
                         "Communs", "DBSCAN seul", "TP 1 seul"],
            "ids": [str(noise_ids), str(tp1_union),
                    str(inter), str(only_db), str(only_tp1)],
            "n": [len(noise_ids), len(tp1_union),
                  len(inter), len(only_db), len(only_tp1)],
        }
    )
    st.dataframe(cmp, use_container_width=True, hide_index=True)

    st.markdown(
        """
Comparaison et interprétation :

- Oui, DBSCAN détecte bien des outliers : ce sont les points de bruit (label -1), isolés dans
  les zones de faible densité.
- Le recouvrement avec le TP 1 est partiel, et c'est attendu. Le TP 1 détecte des outliers
  univariés (valeurs extrêmes d'une seule variable, revenu ou solde_compte), alors que DBSCAN
  raisonne en multivarié sur l'ensemble des variables : il signale des combinaisons atypiques
  même quand chaque variable prise isolément reste dans la norme.
- De plus, prepare_clients() a winsorisé revenu et solde_compte (valeurs extrêmes ramenées aux
  bornes) avant le clustering : les outliers purement univariés de ces deux colonnes sont donc
  atténués, ce qui explique que certains outliers du TP 1 ne ressortent plus comme bruit
  DBSCAN. Inversement, des clients atypiques sur d'autres variables apparaissent comme bruit
  sans avoir été des outliers univariés en TP 1.
"""
    )

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from utils import prepare_clients, load_iris_df

st.set_page_config(page_title="Clustering", layout="wide")
st.title("Clustering par K-Moyennes et CAH")


dataset = st.selectbox("Jeu de données", ["Iris", "Clients banque"])

if dataset == "Iris":
    df = load_iris_df()
    label_col = "species"
    feat_cols = [c for c in df.columns if c != label_col]
    true_labels = df[label_col]
else:
    df = prepare_clients()
    id_col = next((c for c in df.columns if c.lower() == "id"), None)
    feat_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != id_col
    ]
    label_col = None
    true_labels = None

X = df[feat_cols].to_numpy(dtype=float)

pca_plane = PCA(n_components=2).fit(X)
acp = pca_plane.transform(X)


espace = st.radio(
    "Espace de travail (sur lequel le clustering est calculé)",
    ["Données standardisées originales", "ACP (2D)", "Isomap (2D)"],
    horizontal=True,
)

if espace == "Données standardisées originales":
    X_work = X
    work_cols = feat_cols
elif espace == "ACP (2D)":
    X_work = acp
    work_cols = ["PC1", "PC2"]
else:
    X_work = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
    work_cols = ["Iso1", "Iso2"]

st.caption(
    f"{dataset} — {X.shape[0]} individus. Clustering calculé dans l'espace "
    f"« {espace} » ({X_work.shape[1]} dimensions). "
    "Les nuages sont toujours affichés dans le plan ACP pour rester comparables."
)


def centroids_in_acp(coords_acp, labels):
    return np.array([coords_acp[labels == c].mean(axis=0) for c in np.unique(labels)])


def plot_clusters(coords_acp, labels, title, centers_acp=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords_acp[:, 0], coords_acp[:, 1], c=labels, cmap="tab10",
               alpha=0.8, s=25)
    if centers_acp is not None:
        ax.scatter(centers_acp[:, 0], centers_acp[:, 1], marker="X", s=220,
                   c="black", edgecolors="white", linewidths=1.5, label="centroïdes")
        ax.legend()
    ax.set_xlabel(f"PC1 ({pca_plane.explained_variance_ratio_[0] * 100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca_plane.explained_variance_ratio_[1] * 100:.1f} %)")
    ax.set_title(title)
    return fig


st.divider()

st.header("3. K-Moyennes")
c1, c2 = st.columns(2)
with c1:
    k = st.slider("Nombre de clusters k", min_value=2, max_value=10, value=3)
with c2:
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)

km = KMeans(n_clusters=k, random_state=int(random_state), n_init=10)
km_labels = km.fit_predict(X_work)

# (c) Centroïdes
st.subheader("Centroïdes (cluster_centers_)")
centers_df = pd.DataFrame(
    km.cluster_centers_,
    columns=work_cols,
    index=[f"cluster {i}" for i in range(k)],
).round(3)
st.dataframe(centers_df, use_container_width=True)

col_v, col_i = st.columns([1.4, 1])
with col_v:
    centers_acp = centroids_in_acp(acp, km_labels)
    fig = plot_clusters(acp, km_labels, f"K-Moyennes (k = {k}) — plan ACP", centers_acp)
    st.pyplot(fig)
with col_i:
    # (f) inertie
    st.metric("inertia_ (inertie intra-cluster)", f"{km.inertia_:.2f}")
    st.caption(
        "L'inertie est la somme des distances au carré de chaque point à son centroïde. "
        "Elle décroît mécaniquement quand k augmente ; on la lit avec la méthode du coude."
    )

st.subheader("Effet de la graine aléatoire")
seeds = list(range(int(random_state), int(random_state) + 5))
inerties_1 = [
    KMeans(n_clusters=k, random_state=s, n_init=1).fit(X_work).inertia_
    for s in seeds
]
inerties_10 = [
    KMeans(n_clusters=k, random_state=s, n_init=10).fit(X_work).inertia_
    for s in seeds
]
comp_seed = pd.DataFrame(
    {"random_state": seeds,
     "inertia (n_init=1)": np.round(inerties_1, 2),
     "inertia (n_init=10)": np.round(inerties_10, 2)}
)
st.dataframe(comp_seed, use_container_width=True, hide_index=True)
st.markdown(
    """
Constat : avec n_init=1, l'inertie change d'une graine à l'autre — les partitions ne sont
pas identiques. Phénomène : K-Moyennes part de centres initiaux aléatoires et converge vers
un minimum local de l'inertie ; selon l'initialisation, il tombe dans des minima différents.
L'algorithme est donc sensible à l'initialisation.

Solution proposée par sklearn : le paramètre n_init relance l'algorithme plusieurs fois avec
des initialisations différentes (par défaut via k-means++, qui éloigne les centres de départ)
et conserve la meilleure partition (inertie la plus faible). Avec n_init=10, les valeurs
ci-dessus se stabilisent : le résultat devient quasi reproductible.
"""
)

st.divider()

st.header("4. Classification ascendante hiérarchique (CAH)")
linkage_method = st.radio(
    "Critère de linkage",
    ["single (saut minimum)", "complete (saut maximum)", "average (lien moyen)", "ward"],
    horizontal=True,
)
method = linkage_method.split()[0]

Z = linkage(X_work, method=method)
dist_max = float(Z[:, 2].max())

col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    tronquer = st.checkbox("Tronquer le dendrogramme (truncate_mode='lastp')", value=True)
with col_opt2:
    seuil = st.slider(
        "Seuil de coupure (hauteur)",
        min_value=0.0, max_value=round(dist_max, 2),
        value=round(0.6 * dist_max, 2), step=round(dist_max / 100, 3),
    )

fig, ax = plt.subplots(figsize=(10, 4.5))
dendro_kwargs = dict(color_threshold=seuil, ax=ax)
if tronquer:
    dendro_kwargs.update(truncate_mode="lastp", p=30, show_contracted=True)
dendrogram(Z, **dendro_kwargs)
ax.axhline(seuil, color="red", lw=1.5, ls="--")
ax.set_title(f"Dendrogramme — linkage = {method}")
ax.set_xlabel("Individus (ou groupes contractés)")
ax.set_ylabel("Distance de fusion")
st.pyplot(fig)

cah_labels = fcluster(Z, t=seuil, criterion="distance")
n_clusters_cah = len(np.unique(cah_labels))
st.metric("Nombre de clusters obtenus à ce seuil", n_clusters_cah)

fig = plot_clusters(
    acp, cah_labels, f"CAH (linkage = {method}, {n_clusters_cah} clusters) — plan ACP"
)
st.pyplot(fig)

st.divider()

st.header("5. Comparaison des quatre critères de linkage")
if dataset == "Iris":
    st.markdown(
        "Partitions obtenues en coupant chaque dendrogramme pour forcer 3 clusters "
        "(`fcluster(..., criterion='maxclust', t=3`) :"
    )
    from sklearn.metrics import adjusted_rand_score

    y_true = pd.Categorical(true_labels).codes
    rows = []
    figs = {}
    for m in ["single", "complete", "average", "ward"]:
        Zm = linkage(X_work, method=m)
        lab = fcluster(Zm, t=3, criterion="maxclust")
        ari = adjusted_rand_score(y_true, lab)
        sizes = sorted((int(s) for s in np.bincount(lab)[1:]), reverse=True)
        rows.append({"linkage": m, "ARI vs espèces": round(ari, 3),
                     "tailles des clusters": str(sizes)})
        figs[m] = lab
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    cols = st.columns(4)
    for col, m in zip(cols, ["single", "complete", "average", "ward"]):
        with col:
            fig = plot_clusters(acp, figs[m], m)
            st.pyplot(fig)

    st.markdown(
        """
Lecture (ARI = 1 si la partition recouvre exactement les trois espèces) :

- ward et average reproduisent en général le mieux les trois espèces : leurs dendrogrammes
  montrent trois branches nettes et équilibrées, et la visualisation ACP donne trois clusters
  compacts proches des vraies espèces (ARI le plus élevé).
- complete donne aussi des clusters compacts (il limite le diamètre des groupes), mais coupe
  parfois mal la frontière versicolor / virginica.
- single produit l'effet de chaîne (chaining) : il fusionne par le point le plus proche, ce
  qui agrège les individus de proche en proche en un long « ruban ». Sur le dendrogramme cela
  se voit par des fusions à très faible hauteur qui empilent les points un à un, et sur l'ACP
  par un gros cluster qui aspire presque tout, laissant des singletons.

Synthèse : single = effet de chaîne (clusters étirés, déséquilibrés) ; ward (et complete)
= clusters compacts. Pour Iris, ward est le meilleur compromis.
"""
    )
else:
    st.info(
        "La comparaison chiffrée aux vraies classes n'est disponible que pour Iris "
        "(jeu étiqueté). Changez les critères de linkage ci-dessus pour comparer "
        "visuellement les dendrogrammes et les projections ACP sur Clients banque."
    )

st.divider()

st.header("6. Tables de contingence clusters vs vraies espèces")
if dataset == "Iris":
    c_km, c_cah = st.columns(2)
    with c_km:
        st.subheader("K-Moyennes")
        ct_km = pd.crosstab(
            pd.Series(km_labels, name="cluster K-Moyennes"),
            pd.Series(true_labels.to_numpy(), name="espèce"),
        )
        st.dataframe(ct_km, use_container_width=True)
    with c_cah:
        st.subheader("CAH")
        ct_cah = pd.crosstab(
            pd.Series(cah_labels, name="cluster CAH"),
            pd.Series(true_labels.to_numpy(), name="espèce"),
        )
        st.dataframe(ct_cah, use_container_width=True)
    st.caption(
        "Une espèce bien retrouvée se concentre sur une seule ligne (un cluster). "
        "Les colonnes étalées sur plusieurs clusters signalent une espèce mal séparée "
        "(typiquement versicolor / virginica)."
    )
else:
    st.info(
        "Tables de contingence disponibles uniquement pour Iris : Clients banque "
        "n'a pas de vraies classes de référence."
    )

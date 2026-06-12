import streamlit as st
from utils import load_data

st.set_page_config(page_title="Accueil", layout="wide")
st.title("Prétraitement de données — Clients")

# --- 1. Contexte du jeu de données & du projet --------------------------------
st.markdown(
    """
## Contexte

Ce projet porte sur un jeu de données **clients** d'une enseigne commerciale.
Chaque ligne décrit un client à travers des variables **qualitatives** (genre,
lieu de résidence, catégorie de fidélité) et **quantitatives** (âge, revenu,
nombre d'enfants, satisfaction, dépenses mensuelles, ancienneté, solde de compte).

Comme toute donnée réelle, ce fichier est **imparfait** : valeurs manquantes,
valeurs aberrantes (*outliers*), distributions asymétriques et variables exprimées
dans des **échelles très hétérogènes** (des euros face à un nombre d'enfants).

## Objectif du projet

Construire, étape par étape, un **pipeline de prétraitement** rendant ces données
exploitables par des algorithmes de *data mining*. Chaque page de l'application
traite une étape du nettoyage et documente les **choix méthodologiques** retenus.
"""
)

# Petit encart dynamique sur les dimensions du jeu de données
try:
    df = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients (lignes)", df.shape[0])
    c2.metric("Variables (colonnes)", df.shape[1])
    c3.metric("Valeurs manquantes", int(df.isna().sum().sum()))
except FileNotFoundError as e:
    st.warning(str(e))

st.divider()

# --- 2. Schéma du pipeline ----------------------------------------------------
st.subheader("Pipeline de prétraitement")

st.graphviz_chart(
    """
    digraph pipeline {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#e8f0fe",
              fontname="Helvetica", color="#4285f4"];

        exploration   [label="1. Exploration\n(types, stats, structure)"];
        nan           [label="2. Traitement des NaN\n(imputation / suppression)"];
        outliers      [label="3. Traitement des outliers\n(IQR, z-score, MAD)"];
        distribution  [label="4. Étude de distribution\n(asymétrie, discrétisation)"];
        normalisation [label="5. Normalisation\n(min-max, z-score, robuste)"];

        exploration -> nan -> outliers -> distribution -> normalisation;
    }
    """
)

st.caption(
    "Chaque étape s'appuie sur la précédente : on ne peut traiter les outliers "
    "qu'une fois les valeurs manquantes gérées, ni normaliser proprement sans "
    "connaître la forme des distributions."
)

st.divider()

# --- 3. Conclusion : variables conservées et nettoyage ------------------------
st.subheader("Conclusion — variables conservées et nettoyage")

st.markdown(
    """
### Variables conservées

| Variable | Type | Décision |
|---|---|---|
| `id` | identifiant | Conservée comme **clé** (jamais utilisée comme variable explicative). |
| `genre`, `residence` | qualitatif nominal | Conservées (à encoder en *one-hot* avant modélisation). |
| `categorie` | qualitatif ordinal | Conservée (`Standard < Premium`, encodage ordinal). |
| `age` | quantitatif | Conservée — distribution régulière, peu d'outliers. |
| `revenu` | quantitatif | Conservée — variable clé, mais **étalée à droite** et avec outliers. |
| `depenses_mensuelles` | quantitatif | Conservée après **transformation log** (forte asymétrie positive). |
| `solde_compte` | quantitatif | Conservée — grande amplitude, à **normaliser** impérativement. |
| `satisfaction` | quantitatif | Conservée — échelle bornée, sert de cible potentielle. |
| `nb_enfants`, `anciennete_mois` | quantitatif | Conservées — faible amplitude, peu problématiques. |

Aucune variable n'a été **supprimée** : toutes apportent de l'information métier.
Le nettoyage a porté sur la **qualité** des valeurs, pas sur le retrait de colonnes.

### Comment et pourquoi les avons-nous nettoyées ?

1. **Valeurs manquantes** — imputées plutôt que les lignes supprimées (pour ne pas
   perdre d'effectif) : médiane pour les variables asymétriques, et imputation
   plus fine (KNN / régression) là où des corrélations le justifiaient.
2. **Outliers** — détectés par **IQR** et **z-score modifié (MAD)**, plus robustes
   que le z-score classique sujet au *masquage*. Selon les cas : conservés,
   *winsorisés* (ramenés aux bornes) ou atténués par **transformation log** plutôt
   que purement supprimés, pour préserver l'information des clients extrêmes réels.
3. **Distributions** — les variables de type montant (`revenu`,
   `depenses_mensuelles`) sont **étalées à droite** ; une **transformation log**
   réduit leur asymétrie et les rapproche de la normalité.
4. **Normalisation** — indispensable car les échelles sont hétérogènes. On retient
   le **RobustScaler** (médiane / IQR) pour les variables porteuses d'outliers comme
   `revenu`, afin que les distances entre clients ne soient pas dictées par la
   simple unité de mesure.

> **Bilan** : un jeu de données complet, débarrassé des aberrations extrêmes,
> aux distributions assainies et aux variables ramenées sur une échelle commune —
> prêt pour le *clustering* (k-means), la classification (k-NN) ou une ACP.
"""
)

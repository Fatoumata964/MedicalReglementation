import spacy
import scispacy
from bioSentVec import embed_disease
from utils import preprocess_text, process
import pickle
import numpy as np
import csv
import pandas as pd

#df = pd.read_csv('./data/processed/dataEMAfr.csv')
# Chargement du modèle Spacy pour la reconnaissance d'entités nommées
sci_nlp = spacy.load('en_ner_bc5cdr_md')

# Fonction pour extraire les maladies à partir d'un texte donné
def extract_disease(text):
    # Utilisation du modèle Spacy pour identifier les entités nommées
    indications = sci_nlp(text)
    # Extraction des entités identifiées comme étant des maladies
    results = [ent.text for ent in indications.ents if ent.label_ == 'DISEASE']
    return results

# Fonction pour entraîner le modèle
def train(df):
    # Création d'une colonne combinant deux colonnes existantes du dataframe
    df['Combined'] = df[['Espace thérapeutique', 'État/indication']].agg(': '.join, axis=1)
    # Prétraitement du texte combiné
    df['Combined'] = df['Combined'].apply(preprocess_text)
    df['Substance active'] = df['Substance active'].apply(process)
    # Extraction des maladies à partir du texte prétraité
    df['Diseases'] = df['Combined'].apply(lambda x: extract_disease(x))
    # Suppression des lignes où aucune maladie n'a été extraite
    df1 = df[df.astype(str)['Diseases'] != '[]']
    # Application de l'embedding aux substances actives et aux maladies
    df1['Subs_emb'] = df1['Substance active'].apply(lambda x: embed_disease(x))
    df1['Diseases_emb'] = df1['Diseases'].apply(lambda x: embed_disease(x))
    # Concaténation des embeddings pour former la matrice d'embedding
    embedding_subs = np.vstack(df1['Subs_emb'].values)
    embedding_DIS = np.vstack(df1['Diseases_emb'].values)
    embedding_matrix = np.hstack((embedding_DIS, embedding_subs))  
    # Chargement du modèle de clustering pré-entraîné
    with open("./models/clustering_model.pkl", 'rb') as f:
        kmeans = pickle.load(f)
    # Entraînement du modèle de clustering sur la matrice d'embedding
    kmeans.fit(embedding_matrix)
    # Attribution des étiquettes de cluster aux données
    cluster_labels = kmeans.labels_
    df1['cluster_labels'] = cluster_labels
    # Écrire le DataFrame dans un fichier CSV en utilisant la méthode to_csv de Pandas
    df1.to_csv('./data/processed/data_cluster.csv', index=False, encoding='utf-8')
    # Retourne le dataframe avec les étiquettes de cluster attribuées
    return df1

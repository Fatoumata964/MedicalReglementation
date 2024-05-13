import faiss
import pandas as pd
import numpy as np
from bioSentVec import *

# Fonction pour indexer les données textuelles et créer un index FAISS
def indexation(df):
    # Initialiser une liste pour stocker les vecteurs d'embedding de chaque médicament
    embeddings = []

    # Parcourir chaque ligne du dataframe
    for index, row in df.iterrows():
        # Convertir les valeurs des colonnes en chaînes de caractères et les concaténer
        text_data = str(row['Substance active']) + ' ' + str(row['Espace thérapeutique']) + ' ' + str(row["Statut d'autorisation"]) + ' ' + str(row['usage_df1']) + ' ' + str(row['risque'])
        
        # Appliquer la fonction embed_disease à chaque texte pour obtenir le vecteur d'embedding
        embedding = embed_disease(text_data)
        
        # Ajouter le vecteur d'embedding à la liste
        embeddings.append(embedding)

    # Convertir la liste de vecteurs en un tableau numpy
    X = np.array(embeddings)

    # Initialiser l'index FAISS
    d = X.shape[1]  # Assurez-vous que d correspond à la dimension de vos vecteurs d'embedding
    index = faiss.IndexFlatL2(d)

    X = X.reshape(-1, d)
    # Ajouter les vecteurs d'embedding à l'index FAISS
    index.add(X)

    # Créer la carte de correspondance entre le nom du médicament et son index dans l'index FAISS
    drug_to_index_map = {drug_name: index for index, drug_name in enumerate(df['Nom du médicament'])}
    # Retourner l'index FAISS, la carte de correspondance et les données intégrées
    return index, drug_to_index_map, X

# Fonction pour rechercher des médicaments similaires à un médicament donné à l'aide de l'index FAISS
def faiss_search_similar_medications(subs, predicted_cluster, df, k):
    # Appeler la fonction d'indexation pour créer l'index FAISS
    faiss_index, drug_to_index_map, X = indexation(df)
    drugname = df["Nom du médicament"][df["Substance active"] == subs].iloc[0]
    
    # Obtenir l'index du médicament spécifié dans l'index FAISS
    drug_index = drug_to_index_map[drugname]
    
    # Rechercher les k médicaments les plus similaires au médicament spécifié
    D, I = faiss_index.search(X[drug_index].reshape(1, -1), k)
    
    # Initialiser une liste pour stocker les médicaments similaires
    similar_medications = []
  
    # Parcourir les indices des médicaments similaires et leurs clusters prédits
    for neighbor_index, neighbor_cluster in zip(I[0], predicted_cluster):

        # Récupérer les informations sur le médicament voisin
        similar_medication_info = df.iloc[neighbor_index]

        # Vérifier si le cluster du médicament voisin correspond au cluster prédit
        if neighbor_cluster == predicted_cluster:
            # Ajouter les informations sur le médicament voisin à la liste des médicaments similaires
            similar_medications.append(similar_medication_info)
    
    # Retourner la liste des médicaments similaires
    return similar_medications

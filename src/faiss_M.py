import faiss
import pandas as pd
from bioSentVec import *

# Fonction pour indexer les données textuelles et créer un index FAISS
def indexation(text_data, df):
    # Convertir les données textuelles en type str
    text_data = text_data.astype(str)
    
    # Intégrer les données textuelles
    X = embed_disease(text_data)
    
    # Convertir les données intégrées en type 'float32'
    X = X.astype('float32')
    
    # Créer un index FAISS de type IndexFlatL2
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    
    # Créer une carte de correspondance entre le nom du médicament et son index dans l'index FAISS
    drug_to_index_map = {drug_name: index for index, drug_name in enumerate(df['Nom du médicament'])}
    
    # Retourner l'index FAISS, la carte de correspondance et les données intégrées
    return index, drug_to_index_map, X

# Fonction pour rechercher des médicaments similaires à un médicament donné à l'aide de l'index FAISS
def faiss_search_similar_medications(subs, predicted_cluster, text_data, df, k):
    # Appeler la fonction d'indexation pour créer l'index FAISS
    faiss_index, drug_to_index_map, X = indexation(text_data, df)
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

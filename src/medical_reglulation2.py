import time
import uvicorn
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from faiss_M import faiss_search_similar_medications
from llm import llm
from ner import train
import pickle
import os
import pandas as pd
import numpy as np
import sent2vec

app = FastAPI()
cc =  ColabCode(port=8002, code=False)
df = pd.read_csv("./data/processed/dataEMAfr.csv")
mistral_llm = llm()

# Initialisation du modèle Sent2Vec
model = sent2vec.Sent2vecModel()
try:
      # Chargement du modèle depuis le chemin spécifié
        model.load_model("/content/drive/MyDrive/stage/hh/Medical_Reglementation/models/biosentvec.crdownload")
except Exception as e:
      # Gestion des erreurs lors du chargement du modèle
      print(e)

# Define a Pydantic model to validate the input
class TextInput(BaseModel):
    # text: str = "Povidone, amidon prégélatinisé, carboxyméthylamidon sodique (type A), talc, stéarate de magnésium"
    text: str = "aflibercept"

def extract_regulation(drug):
    '''Extraction de la réglementation du médicament donné'''

    # Entraînement d'un nouveau DataFrame pour le clustering
    new_df = train(df)
    print(new_df[0:1])
    
    # Intégration du texte du médicament et d'une phrase représentative de la maladie
    embedded_drug = model.embed_sentence(drug)
    disease = new_df["Diseases"][new_df["Substance active"] == drug].iloc[0]
    embedded_disease = model.embed_sentence(str(disease))
    
    # Création de la matrice d'embedding en concaténant les embeddings du médicament et de la maladie
    embedding_mat = np.hstack((embedded_disease, embedded_drug))
    
    print("Drug embedded")
    
    # Chargement du modèle de clustering à partir du fichier pickle
    with open("./models/clustering_model.pkl", 'rb') as f:
        kmeans = pickle.load(f)
    
    # Prédiction du cluster auquel appartient le médicament
    y = kmeans.predict(embedding_mat)
    print(y)
    
    # Recherche de médicaments similaires dans le même cluster
    df_clus = new_df[new_df['cluster_labels'] == y[0]]
    # Recherche de médicaments similaires dans le même cluster
    similar_medications_in_cluster = faiss_search_similar_medications(drug, df_clus, 10)
    print(similar_medications_in_cluster)

    # Construction du prompt pour la génération de la réglementation
    prompt = f'[INST] Tu es un assistant médical, un assistant aimable et utile. Ton rôle est de donner une réglementation pour un médicament donné en te basant sur les donnees fournis. Sois le plus précis et fiable possible. Crée une réglementation détaillée pour l\'utilisation du {drug} en France, ne parles que du {drug} ,en te basant sur les médicaments suivants :'
    
    med = similar_medications_in_cluster[['Substance active', 'Espace thérapeutique', "Statut d'autorisation", 'usage_df1', 'risque']].to_string(index=False)
    prompt += med
    
    last_questions = ''  # Variable pour stocker les dernières questions posées

    prompts = [
        f"## Donne l'Encadré pour la notice de {drug}",
        f"## Que contient cette notice ?",
        f"## Qu'est-ce que {drug} et comment l'utiliser ?",
        f"## Quelles sont les informations à connaître avant de prendre {drug} ?",
        f"## Comment prendre {drug} ?",
        f"## Quels sont les effets indésirables éventuels de {drug} ?",
        f"## Comment conserver {drug} ?",
        f"## Contenu de l'emballage et autres informations concernant {drug}"
    ]

    max_length = 512
    
    for prompt_sec in prompts:

        last_questions += f'\n{prompt_sec}'
        prompt_secw = prompt if prompts[0] else '' + prompt_sec.format(drug=drug) + f'Voici les dernières questions posées:{last_questions}'
        chunks = [prompt_secw[i:i+max_length] for i in range(0, len(prompt_secw), max_length)]

        response = [mistral_llm(chunk) for chunk in chunks]
        time.sleep(2)

    full_response = ''.join(response)

    # Retour du texte généré
    return full_response
    
@app.get('/')
def index():
    """
    Default endpoint for the API.
    """
    return {
        "version": "0.1.0",
        "documentation": "/docs"
    }

# Define a route that accepts POST requests with JSON data containing the text
@app.post("/apiv1/regulation/get-regulation")
async def get_regulation(drug: TextInput):
    # You can perform text processing here
    start_time = time.time()
    print(drug)
    regulation_text = extract_regulation(drug.text)
    # stopping the timer
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    # formatting the elapsed time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(regulation_text)
    # Return the processed text as a response
    return {"regulation": regulation_text}


if __name__ == '__main__':
    cc.run_app(app=app)

import time
import uvicorn
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from faiss_M import faiss_search_similar_medications
from llm import llm
from utils import process
import pickle
import os
import pandas as pd
import numpy as np
import sent2vec

app = FastAPI()
cc =  ColabCode(port=8002, code=False)
df = pd.read_csv("./data/processed/data_cluster.csv")
mistral_llm = llm()

# Initialisation du modèle Sent2Vec
model = sent2vec.Sent2vecModel()
try:
      # Chargement du modèle depuis le chemin spécifié
        model.load_model("./models/biosentvec.crdownload")
except Exception as e:
      # Gestion des erreurs lors du chargement du modèle
      print(e)

# Define a Pydantic model to validate the input
class TextInput(BaseModel):
    # text: str = "Povidone, amidon prégélatinisé, carboxyméthylamidon sodique (type A), talc, stéarate de magnésium"
    text: str = "aflibercept"

def extract_regulation(drug):
    '''Extraction de la réglementation du médicament donné'''
    drug = process(drug)
    print(df[0:1])
    
    # Intégration du texte du médicament et d'une phrase représentative de la maladie
    embedded_drug = model.embed_sentence(drug)
    disease = df["Diseases"][df["Substance active"] == drug].iloc[0]
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
    df_clus = df[df['cluster_labels'] == y[0]]
    # Recherche de médicaments similaires dans le même cluster
    similar_medications_in_cluster = faiss_search_similar_medications(drug, df_clus, 2)
    print(similar_medications_in_cluster)

    # Construction du prompt pour la génération de la réglementation
    prompt = f'[INST] Tu es un assistant médical, un assistant aimable et utile. Ton rôle est de donner une réglementation pour un médicament donné en te basant sur les donnees fournis. Sois le plus précis et fiable possible. Crée une réglementation détaillée pour l\'utilisation du {drug} en France, ne parles que du {drug} ,en te basant sur les médicaments suivants :'
    
    med = similar_medications_in_cluster[['Substance active', 'Espace thérapeutique', "Statut d'autorisation", 'usage_df1', 'risque']].to_string(index=False)
    prompt += med
    
    # Ajout d'informations supplémentaires en fonction du cluster prédit
    if y == 0:
      prompt += ''' Les médicaments dans ce groupe traitent le plus souvent les types de cancer, en particulier les cancers du sein et des poumons, ainsi que les fractures. 
      Pour les médicaments traitant le cancer, la veille a montré le bénéfice significatif qu'ils apportent aux patients en termes de survie, de qualité de vie ou d'autres critères pertinents. 
      Ils doivent obligatoirement obtenir une autorisation de mise sur le marché. Vérifiez le statut d'autorisation dans le contexte pour cela. '''
    elif y == 1:
      prompt += ''' Les médicaments dans ce groupe sont principalement destinés au traitement des infections et de la fibrose. Il est essentiel de sensibiliser le patient à l'importance 
      de l'utilisation rationnelle des antibiotiques pour lutter contre les infections. De plus, pour les médicaments visant à traiter la fibrose, il est crucial qu'ils présentent un 
      profil de sécurité acceptable, garantissant ainsi la santé et le bien-être des patients. '''
    elif y == 2:
      prompt += ''' Les médicaments dans ce groupe traitent le plus souvent l'hypertension essentielle et l'insuffisance rénale. Les médicaments contre l'hypertension doivent être 
      sûrs et efficaces pour les populations à risque, telles que les personnes âgées, les personnes souffrant d'autres maladies et les femmes enceintes ou allaitantes. 
      Pour les médicaments contre l'insuffisance rénale, il est crucial d'informer les patients de l'importance de la surveillance de la fonction rénale pendant le traitement, afin de garantir des résultats optimaux. '''
    else :
      prompt += ''' 'Les médicaments dans ce groupe sont principalement utilisés pour le traitement des cancers, notamment chez les adultes atteints. La réglementation des médicaments 
      contre le cancer doit prendre en compte la complexité de la maladie, ses différentes formes et ses stades de progression. De plus, les effets indésirables associés aux traitements 
      contre le cancer doivent être rigoureusement documentés et gérés de manière appropriée pour assurer la sécurité des patients. En parallèle, les soins des patients atteints de maladies 
      telles que le psoriasis nécessitent une attention particulière. Il est crucial de surveiller de près les effets indésirables des médicaments utilisés pour traiter le psoriasis, 
      car ils peuvent avoir un impact significatif sur la qualité de vie des patients. De même, la réglementation des médicaments contre la maladie de Parkinson doit être adaptée aux caractéristiques
       uniques de cette maladie neurodégénérative, telles que la progression progressive de la maladie et la variabilité des symptômes d'un patient à l'autre. Il est essentiel que les traitements disponibles 
       pour la maladie de Parkinson soient efficaces et sûrs, permettant ainsi d'améliorer la qualité de vie des patients et de leur offrir un soulagement optimal des symptômes.'''
          
    prompt += '''
    Assures-toi d'inclure explicitement le Statut d'autorisation, la Date de révision, les dosages recommandés, les incompatibilités, les mises en garde spéciales et précautions d'emploi, ainsi que les contre-indications et l'URL vers le site EMA du medicament en te basant sur le context donné, sans donner d'informaions sur les médicaments ci-dessus.

    Contexte:
    {similar_medications_in_cluster, df}

     Le format de sortie doit être en Markdown :
     Le titre principal doit commencer par #.
     Les sous-titres doivent commencer par ##.
     Les listes doivent commencer par *.
     '''

    # Découpage du prompt en chunks pour respecter la limite de longueur
    max_length = 512
    prompt_chunks = [prompt[i:i+max_length] for i in range(0, len(prompt), max_length)]

    # Génération du texte réglementaire à partir du prompt en utilisant le modèle de langage mixte
    generated_text_chunks = []
    for prompt_chunk in prompt_chunks:
        output = mistral_llm(prompt_chunk)
        generated_text_chunks.append(output)

    generated_text = ''.join(generated_text_chunks)
    generated_text = '\n'.join([p for p in generated_text.split('\n')[1:] if len(p) > 0])

    # Retour du texte généré
    return generated_text
    
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
    regulation_text = extract_regulation(drug.text.lower())
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

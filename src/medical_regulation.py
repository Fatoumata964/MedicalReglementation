import time
import uvicorn
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from faiss_M import *
from bioSentVec import *
from llm import *
from ner import *
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
    
    # Création d'une liste de données textuelles pour la recherche de médicaments similaires
    text_data = new_df['Substance active'] + ' ' + new_df['Espace thérapeutique'] + ' ' + new_df["Statut d'autorisation"] + ' ' + new_df['usage_df1'] + ' ' + new_df['risque']
    
    # Recherche de médicaments similaires dans le même cluster
    similar_medications_in_cluster = faiss_search_similar_medications(drug, y, text_data, new_df, 10)
    
    # Construction du prompt pour la génération de la réglementation
    prompt = f'[INST] Tu es un assistant médical, un assistant aimable et utile. Ton rôle est de donner une réglementation pour un médicament donné en te basant sur les donnees fournis. Sois le plus précis et fiable possible. Crée une réglementation détaillée pour l\'utilisation du {drug} en France, ne parles que du {drug} ,en te basant sur les médicaments suivants :'
    med = ''
    for drug_info in similar_medications_in_cluster:
        med += drug_info["Date de révision"] + ' ' + drug_info["Statut d'autorisation"] + ' ' + drug_info["Espace thérapeutique"] + ' ' + drug_info["État/indication"] + ' ' + drug_info["usage_df1"] + ' ' + drug_info["risque"] + ' ' + drug_info["URL"]
    prompt += med
    
    # Ajout d'informations supplémentaires en fonction du cluster prédit
    if y == 0:
      prompt += ''' Les medicaments dans ce groupe traite le plus souvent les types de cancer seins et poumons et l'infarctus myocarde.
      Pour les medicaments traitant le cancer, veille a montré le bénéfice significatif qu'il apporte aux patients en termes de survie, de qualité de vie ou d'autres critères pertinents.
      Ils doivent obligatoirement obtenir une autorisation de mise sur le marché, verifie le statut d'autorisation dans le contexte pour cela.
      La réglementation d'un médicament contre l'infarctus du myocarde doit tenir compte des caractéristiques spécifiques du médicament et de la population de patients qu'il est censé traiter.'''
    elif y == 1:
      prompt += ''' Les medicaments dans ce groupe traite le plus souvent les infections et fibrose. Veille a sensibiliser le patient à l'importance de l'utilisation rationnelle des antibiotiques.
      Pour les medicaments contre la fibrose, ils doivent avoir un profil de sécurité acceptable. '''
    elif y == 2:
      prompt += ''' Les medicaments dans ce groupe traite le plus souvent l'hypertension et l'insuffisance renale. Les médicaments contre l'hypertension doivent être sûrs et efficaces pour les populations à risque,
      telles que les personnes âgées, les personnes souffrant d'autres maladies et les femmes enceintes ou allaitantes. Pour les médicaments contre l'insuffisance renale, informer les patients de l'importance de
      la surveillance de la fonction rénale pendant le traitement'''
    else :
      prompt += ''' Les medicaments dans ce groupe traite le plus souvent les cancers, psoriasis et maladie parkinson. Les effets indésirables pour les medicaments contre le psoriasis doivent être bien documentés
      et gérés de manière appropriée. La réglementation des médicaments contre la maladie de Parkinson doit tenir compte des caractéristiques spécifiques de la maladie, telles que:
      la progression progressive de la maladie et la variabilité des symptômes d'un patient à l'autre'''

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

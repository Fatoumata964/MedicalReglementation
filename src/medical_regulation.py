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
    similar_medications_in_cluster = faiss_search_similar_medications(drug, y, new_df, 10)
    
    # Construction du prompt pour la génération de la réglementation
    prompt = f'[INST] Tu es un assistant médical, un assistant aimable et utile. Ton rôle est de donner une réglementation pour un médicament donné en te basant sur les donnees fournis. Sois le plus précis et fiable possible. Crée une réglementation détaillée pour l\'utilisation du {drug} en France, ne parles que du {drug} ,en te basant sur les médicaments suivants :'
    med = ''
    for drug_info in similar_medications_in_cluster:
        med += drug_info["Date de révision"] + ' ' + drug_info["Statut d'autorisation"] + ' ' + drug_info["Espace thérapeutique"] + ' ' + drug_info["État/indication"] + ' ' + drug_info["usage_df1"] + ' ' + drug_info["risque"] + ' ' + drug_info["URL"]
    prompt += med
    
    prompt += '''Tu es un assistant médical, un assistant aimable et utile. Ton rôle est de donner une réglementation pour un médicament donné en te basant sur les donnees fournis. Sois le plus précis et fiable possible. Crée une réglementation détaillée pour l\'utilisation de L'\aflibercept en France, ne parles que de L'aflibercept . En suivant la sructure suivante, remplace drug par aflibercept :
        f"## Donne l'Encadré pour la notice de {drug}",
        f"## Que contient cette notice ?",
        f"## Qu'est-ce que {drug} et comment l'utiliser ?",
        f"## Quelles sont les informations à connaître avant de prendre {drug} ?",
        f"## Comment prendre {drug} ?",
        f"## Quels sont les effets indésirables éventuels de {drug} ?",
        f"## Comment conserver {drug} ?",
        f"## Contenu de l'emballage et autres informations concernant {drug}".
        
       Utilise l'exemple suivant comme guide: 

        Nom du médicament
        PARACÉTAMOL 500 mg, comprimé

        Composition qualitative et quantitative
        Chaque comprimé contient 500 mg de paracétamol.

        Forme pharmaceutique
        Comprimé.

        Classe thérapeutique
        Médicament analgésique et antipyrétique.

        Indications thérapeutiques
        Traitement symptomatique de la fièvre et des douleurs d'intensité légère à modérée, telles que les maux de tête, les douleurs dentaires, les douleurs musculaires, les règles douloureuses et les symptômes du rhume et de la grippe.

        Posologie et mode d'administration
        La posologie recommandée pour les adultes et les enfants de plus de 12 ans est de 1 à 2 comprimés par prise, à renouveler si nécessaire toutes les 4 à 6 heures. Ne pas dépasser 8 comprimés par jour.

        Que contient cette notice ?
        Cette notice contient des informations importantes sur l'utilisation sûre et efficace du médicament PARACÉTAMOL. Il est essentiel de lire attentivement cette notice avant d'utiliser ce médicament et de suivre les instructions fournies par votre médecin ou votre professionnel de la santé.

        Qu'est-ce que le paracétamol et comment l'utiliser ?
        Le paracétamol est un médicament utilisé pour traiter la fièvre et les douleurs légères à modérées. Il agit en réduisant la production de substances dans le cerveau qui provoquent la fièvre et la douleur. Le paracétamol est pris par voie orale sous forme de comprimés, à avaler avec un verre d'eau.

        Quelles sont les informations à connaître avant de prendre le paracétamol ?
        Avant de prendre le paracétamol, informez votre médecin si vous avez des antécédents de problèmes hépatiques, de consommation excessive d'alcool ou si vous prenez d'autres médicaments, en particulier des médicaments contenant du paracétamol ou des anticoagulants. Ne dépassez pas la dose recommandée et ne prenez pas ce médicament pendant une période prolongée sans avis médical.

        Comment prendre le paracétamol ?
        Le paracétamol doit être pris par voie orale avec un verre d'eau. Respectez la posologie recommandée et ne dépassez pas la dose maximale recommandée. Ne prenez pas ce médicament plus longtemps que prescrit sans consulter votre médecin.

        Quels sont les effets indésirables éventuels du paracétamol ?
        Les effets indésirables les plus courants du paracétamol comprennent les réactions allergiques, les nausées, les vomissements et les éruptions cutanées. Des effets indésirables plus graves, tels que les lésions hépatiques, peuvent également survenir en cas de surdosage. Contactez immédiatement votre médecin si vous ressentez des effets indésirables graves.

        Comment conserver le paracétamol ?
        Conservez le paracétamol dans un endroit sec à une température inférieure à 25°C. Gardez-le hors de la portée des enfants et des animaux domestiques. Ne pas utiliser ce médicament après la date de péremption indiquée sur l'emballage.

        Contenu de l'emballage et autres informations concernant le paracétamol
        Chaque boîte contient un nombre déterminé de comprimés de paracétamol. Ne pas utiliser ce médicament si l'emballage est endommagé ou si le médicament a changé de couleur ou d'odeur. Consultez votre pharmacien pour plus d'informations sur le stockage et l'utilisation sûre de ce médicament.'''

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

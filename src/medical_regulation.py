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
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

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
    context = 'Voici des informations pour créer la reglementation de {drug}: '
    med = similar_medications_in_cluster[['Substance active', 'Espace thérapeutique', "Statut d'autorisation", 'usage_df1', 'risque']].to_string(index=False)
    context += med

    example= [{"context": "",
           "drug": drug,
           "encadre": " Veuillez lire attentivement cette notice avant de prendre ce médicament car elle contient des informations importantes pour vous. Vous devez toujours prendre ce médicament en suivant scrupuleusement les informations fournies dans cette notice ou par votre médecin ou votre pharmacien. · Gardez cette notice. Vous pourriez avoir besoin de la relire. · Adressez-vous à votre pharmacien pour tout conseil ou information. · Si vous ressentez un quelconque effet indésirable, parlez-en à votre médecin ou votre pharmacien. Ceci s’applique aussi à tout effet indésirable qui ne serait pas mentionné dans cette notice. Voir rubrique 4.· Vous devez vous adresser à votre médecin si vous ne ressentez aucune amélioration ou si vous vous sentez moins bien.",
           "contenu": "1. Qu'est-ce que {drug} et dans quels cas est-il utilisé ? \n 2. Quelles sont les informations à connaître avant de prendre {drug}? \n 3. Comment prendre {drug}? \n 4. Quels sont les effets indésirables éventuels ?\n 5. Comment conserver {drug} ? \n 6. Contenu de l’emballage et autres informations.",
           "utilite": "Classe pharmacothérapeutique - code ATC : Autres préparations à usage systémique, D10BX (D : Dermatologie). Ce médicament contient du {drug}. Ce médicament est indiqué dans : · Acné inflammatoire de sévérité mineure et moyenne, · Acrodermatite entéropathique.",
           "information": "Ne prenez jamais {drug} : · si vous êtes allergique (hypersensible) au gluconate de {drug} ou à l’un des autres composants contenus dans ce médicament, mentionnés dans la rubrique 6. Avertissements et précautions Adressez-vous à votre médecin ou à votre pharmacien avant de prendre {drug} gélule. Les gélules doivent être prises avec un grand verre d’eau et en position assise afin de limiter le risque de troubles digestifs. La position allongée est à éviter pendant les 30 minutes suivant la prise des gélules.",
           "posologie": "Veillez à toujours prendre ce médicament en suivant exactement les instructions de cette notice ou les indications de votre médecin ou de votre pharmacien. Vérifiez auprès de votre médecin ou de votre pharmacien en cas de doute. Posologie\n A prendre à distance des repas, car le bol alimentaire peut modifier l'absorption du médicament. Acné : La posologie recommandée est de 2 gélules par jour en une seule prise le matin à distance des repas avec un grand verre d'eau.",
           "effet_sec": "Comme tous les médicaments, ce médicament peut provoquer des effets indésirables, mais ils ne surviennent pas systématiquement chez tout le monde. Les effets indésirables suivants ont été rapportés : Rarement (survenant chez moins de 1 patient sur 1 000) : Au cours du traitement, il est possible que surviennent des douleurs de l'estomac et du ventre ; elles sont habituellement de faible intensité et transitoire, ainsi que des nausées, vomissements, constipations ou diarrhées. Très rarement (survenant chez moins de 1 patient sur 10 000) :",
           "conservation": "Tenir ce médicament hors de la vue et de la portée des enfants. N’utilisez pas ce médicament après la date de péremption indiquée sur l’emballage après EXP. La date de péremption fait référence au dernier jour de ce mois. Pas de précautions particulières de conservation.",
           "emballage": "Ce que contient {drug}  Retour en haut de la page· La substance active est : {drug}................................................................................................................................. 15,00 mg Sous forme de gluconate de {drug}..................................................................................... 104,55 mg. Qu’est-ce que {drug} et contenu de l’emballage extérieur. Titulaire de l’autorisation de mise sur le marché  Retour en haut de la page LABCATAL 1198 AVENUE DU DOCTEUR MAURICE DONAT. Exploitant de l’autorisation de mise sur le marché  Retour en haut de la page LABORATOIRE DES GRANIONS. Fabricant LABCATAL. Noms du médicament dans les Etats membres de l'Espace Economique Européen. La dernière date à laquelle cette notice a été révisée est :"

    }]

    example_prompt = PromptTemplate(
        input_variables = ["context", "drug", "encadre", "contenu", "utilite", "information", "posologie", "effet_sec", "conservation", "emballage"],
        template = """
    Tu es un assistant médical, un assistant aimable et utile. Ton rôle est de donner une réglementation pour {drug} en te basant sur les donnees fournis. Sois le plus précis et fiable possible. Crée une réglementation détaillée pour l\'utilisation du {drug} en France, NE parles QUE du {drug} . En utilisant le contexte suivant:

    Context: {context}
    Respecte la structure suivante :
            ## Encadré \n {encadre}\n  ,
            ## Que contient cette notice ? \n{contenu} \n ,
            1- Qu'est-ce que {drug} et comment l'utiliser ? \n{utilite} \n ,
            2- Quelles sont les informations à connaître avant de prendre {drug} ? \n{information}\n ,
            3- Comment prendre {drug} ? \n {posologie}\n ,
            4- Quels sont les effets indésirables éventuels de {drug} \n ?{effet_sec}\n ,
            5- Comment conserver {drug} ?\n {conservation}?\n ,
            6- Contenu de l'emballage et autres informations concernant {drug} \n {emballage}
    """
    )

    # Create the FewshotPromptTemplate
    prompt_template = FewShotPromptTemplate(
        examples=example,
        example_prompt=example_prompt,
        prefix="Voici un exemple d'utilisation correcte:",
        suffix="Notice de: {drug}",
        input_variables=["drug"]
    )
    prompt = prompt_template.format(drug=drug)
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
    cleaned_text = ' '.join(generated_text.split())
    # Retour du texte généré
    return cleaned_text
    
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

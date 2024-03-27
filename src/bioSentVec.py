import numpy as np
import sent2vec

# Initialisation du modèle Sent2Vec
model = sent2vec.Sent2vecModel()
try:
      # Chargement du modèle depuis le chemin spécifié
     model.load_model("./models/biosentvec.crdownload")
except Exception as e:
      # Gestion des erreurs lors du chargement du modèle
    print(e)


# Fonction pour intégrer une phrase en utilisant un modèle d'embedding spécifié
def embed_disease(sentence):
    # Liste pour stocker les phrases intégrées
    embedded_sentences = []
    for el in sentence:
        # Intégration de chaque phrase en utilisant le modèle d'embedding spécifié
        embedded_sentence = model.embed_sentence(el)
        embedded_sentences.append(embedded_sentence)

        # Conversion de la liste en un tableau numpy
        embedded_sentences = np.array(embedded_sentences)

        # Redimensionnement du tableau pour qu'il ait la forme [nombre_de_phrases, taille_de_l'embedding]
        embedded_sentences = embedded_sentences.reshape(embedded_sentences.shape[0], embedded_sentences.shape[2])

        # Retourne les phrases intégrées sous forme de tableau
        return embedded_sentences



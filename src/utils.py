import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
import pickle
import numpy as np
import csv
import re


# Téléchargement des données nécessaires pour NLTK
nltk.download('stopwords')
nltk.download('punkt')


# Définition des mots vides
stop_words = set(stopwords.words('french'))

def process(text):
    preprocessed_text = re.sub(r'\s+', ' ', text).strip()
    
    return preprocessed_text

# Fonction pour prétraiter une phrase
def preprocess_sentence(text):
    # Remplacement de certains caractères spéciaux par des espaces
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    # Conversion du texte en minuscules
    text = text.lower()
    # Tokenisation du texte et suppression des mots vides et de la ponctuation
    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
    # Reconstitution de la phrase prétraitée
    return ' '.join(tokens)
    
# Fonction pour prétraiter un texte
def preprocess_text(text):
    # Appel de la fonction de prétraitement de phrase
    return preprocess_sentence(text)

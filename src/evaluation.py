from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from medical_regulation import extract_regulation
modelf = CrossEncoder("vectara/hallucination_evaluation_model")


# Fonction pour calculer la similarité cosinus entre le texte de réglementation d'un médicament et un autre texte donné
def cosineSimilarity(drug, text2): 
    # Extraction du texte de réglementation pour le médicament donné
    text1 = extract_regulation(drug)
    
    # Création d'une liste contenant les deux textes (text1 et text2)
    documents = [text1, text2]
    
    # Initialisation du CountVectorizer pour convertir les textes en matrices de termes-document
    count_vectorizer = CountVectorizer(stop_words="english")
    count_vectorizer = CountVectorizer()
    
    # Conversion des textes en matrices de termes-document
    sparse_matrix = count_vectorizer.fit_transform(documents)

    # Conversion de la matrice sparse en une matrice dense
    doc_term_matrix = sparse_matrix.todense()
    
    # Création d'un DataFrame pour stocker les matrices de termes-document
    df1 = pd.DataFrame(
        doc_term_matrix,
        columns=count_vectorizer.get_feature_names_out(),
        index=["verite_terrain", "reglementation"],
    )

    # Calcul de la similarité cosinus entre les deux textes
    return cosine_similarity(df1, df1)


# Fonction pour prédire un score de "hallucination" entre le texte de réglementation d'un médicament et un autre texte donné
def score_hallucination(drug, text2):
    # Extraction du texte de réglementation pour le médicament donné
    text1 = extract_regulation(drug)
    
    # Utilisation d'un modèle de prédiction (modelf) pour prédire un score en fournissant une liste contenant les textes text1 et text2
    scores = modelf.predict([
        [text1, text2]
    ])
    
    # Retourne le score prédit
    return scores


if __name__ == "__main__":
    text = "Yesafili est indiqué chez l'adulte pour le traitement de la dégénérescence maculaire liée à l'âge (DMLA) néovasculaire (humide) (voir rubrique 5.1), de la déficience visuelle due à un œdème maculaire secondaire à une occlusion de la veine rétinienne (OVR de branche ou OVR centrale) (voir rubrique 5.1), déficience visuelle due à un œdème maculaire diabétique (OMD) (voir rubrique 5.1), déficience visuelle due à une néovascularisation choroïdienne myope (NVC myopique) (voir rubrique 5.1).Yesafili est disponible sous forme de flacons contenant une solution pour injection intravitréenne (injection dans l'humeur vitrée, le liquide gélatineux situé à l'intérieur de l'œil). Il ne peut être obtenu que sur ordonnance et doit être administré par un médecin expérimenté dans les injections intravitréennes.La sécurité de Yesafili a été évaluée et, sur la base de l'étude réalisée, les effets secondaires du médicament sont considérés comme comparables à ceux du médicament de référence Eylea.\nPour la liste complète des effets secondaires et des restrictions de Yesafili, voir la notice.\nLes effets indésirables les plus couramment observés sous Yesafili (qui peuvent affecter plus d'une personne sur 20) comprennent l'hémorragie conjonctivale (saignement des petits vaisseaux sanguins à la surface de l'œil au site d'injection), l'hémorragie rétinienne (saignement à l'arrière de l'œil). œil), vision réduite, douleur oculaire, décollement du corps vitré (détachement de la substance gélatineuse à l'intérieur de l'œil), cataracte (opacification du cristallin), corps flottants du corps vitré (petites formes sombres se déplaçant dans le champ de vision) et augmentation de la pression intraoculaire (augmentation de la pression à l’intérieur de l’œil), Authorised, 2023-12-04 18:15:00, https://www.ema.europa.eu/en/medicines/human/E..., "
    print("cosineSimilarity: ", cosineSimilarity("aflibercept", text))
    print("score_hallucination: ", score_hallucination("aflibercept", text))
    

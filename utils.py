from sentence_transformers import util
import string
import difflib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Data Processing English language model in spaCy
nlp = spacy.load("en_core_web_sm")

def Title_Essay_Relevancy(model, title,essay):

    title_embedding = model.encode(title)
    essay_embedding = model.encode(essay)

    return util.dot_score(title_embedding, essay_embedding).item()

def Preprocess(essay):
    # Tokenization, lowercasing, stopword removal, punctuation removal, lemmatization
    doc = nlp(essay)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.text not in string.punctuation]

    # Return the preprocessed text as a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def DotProduct_calculate(essay, essay_list):

    # Create TF-IDF vectors for all essays
    vectorizer = TfidfVectorizer()
    if type(essay_list) is str:
        tfidf_matrix = vectorizer.fit_transform([essay_list,essay])
    else:
        tfidf_matrix = vectorizer.fit_transform(essay_list + [essay])
    tfidf_matrix = tfidf_matrix.toarray()

    # Calculate cosine similarity between input essay and existing essays
    similarities = np.dot(tfidf_matrix, tfidf_matrix[-1].T)
    sorted_indices = np.argsort(similarities, axis=0)[::-1]

    return similarities, sorted_indices

def plag_calculate(essay, essay_list):

    max_plag_score = 0
    max_plag_essay = ""

    # Create Vectors and Calculate DotProduct
    similarities, sorted_indices = DotProduct_calculate(essay=essay, essay_list=essay_list)

    # Find the essay with the maximum plagiarism score
    for idx in sorted_indices[1:]:
        if idx < len(essay_list):
            plag_score = similarities[idx]
            if plag_score > max_plag_score:
                max_plag_score = plag_score
                max_plag_essay = essay_list[idx]

    # Adjust the maximum plagiarism score to be between 0 and 1
    max_plag_score = max(0, min(max_plag_score, 1))

    return max_plag_essay, max_plag_score

def Grammar_Spell_Check(gs_model, gs_model_args, essay, cosine_obj):
    doc = nlp(essay)
    sentences = [sent.text for sent in doc.sents]
    corrected_essay = ''
    for sentence in sentences:
        corrected_essay+= gs_model.generate_text(f"grammar: {sentence}", args=gs_model_args).text+' '

    similarities, _ = DotProduct_calculate(essay=essay, essay_list=corrected_essay)
    print('similarities ',similarities)
    return min(similarities), corrected_essay

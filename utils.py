from sentence_transformers import util
import string


def Title_Essay_Relevancy(model, title,essay):

    title_embedding = model.encode(title)
    essay_embedding = model.encode(essay)

    return util.dot_score(title_embedding, essay_embedding).item()

def Preprocess(preprocess_obj, essay):
    # Tokenization, lowercasing, stopword removal, punctuation removal, lemmatization
    doc = preprocess_obj(essay)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.text not in string.punctuation]

    # Return the preprocessed text as a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def plag_calculate(cosine_1g, cosine_2g, cosine_3g, essay:str, db_essay_list:list):
    print('Total essays in DB ',type(db_essay_list))
    # Create list to save result with every essay
    plag_results = []
    for db_essay in db_essay_list:
        sum = 0
        # sum+=cosine_1g.similarity(essay,db_essay)
        # sum+=cosine_2g.similarity(essay,db_essay)
        sum+=cosine_3g.similarity(essay,db_essay)
        plag_results.append(sum)
    print('max(plag_results) ',max(plag_results))
    return max(plag_results)
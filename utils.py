from sentence_transformers import util
import string
import difflib

def Title_Essay_Relevancy(model, title,essay):

    title_embedding = model.encode(title)
    essay_embedding = model.encode(essay)

    return util.dot_score(title_embedding, essay_embedding).item()

def Preprocess(text_process_pipeline, essay):
    # Tokenization, lowercasing, stopword removal, punctuation removal, lemmatization
    doc = text_process_pipeline(essay)
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
        sum+=cosine_2g.similarity(essay,db_essay)
        # sum+=cosine_3g.similarity(essay,db_essay)
        plag_results.append(sum)
        plag = max(plag_results)
    return max(1- plag,0)

def Grammar_Spell_Check(text_process_pipeline, gs_model, gs_model_args, essay, cosine_obj):
    doc = text_process_pipeline(essay)
    sentences = [sent.text for sent in doc.sents]
    corrected_essay = ''
    for sentence in sentences:
        corrected_essay+= gs_model.generate_text(f"grammar: {sentence}", args=gs_model_args).text+' '

    print('\n\n' 'corrected_essay\n',corrected_essay)

    # Split the essays into individual words
    words_corrected_essay = corrected_essay.split()
    words_essay = essay.split()

    # Calculate the differences using difflib
    diff = difflib.ndiff(words_corrected_essay, words_essay)

    # Count the number of differences
    num_differences = sum(1 for _ in diff if not _.startswith(' '))

    # Normalize the number of differences 0 to 1
    max_length = max(len(words_corrected_essay), len(words_essay))
    similarity= 1 - (num_differences / max_length)

    return max(similarity, 0)
    # return cosine_obj.distance(essay,corrected_essay)
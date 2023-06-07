from flask import Flask, request
from sentence_transformers import SentenceTransformer
import spacy
import pymongo
from utils import Preprocess, Title_Essay_Relevancy,plag_calculate,Grammar_Spell_Check
from strsimpy import Cosine
from happytransformer import HappyTextToText, TTSettings

# Intialization
cosine_1g, cosine_2g, cosine_3g = Cosine(1), Cosine(2), Cosine(3)

# MongoDB 
mongo_client = pymongo.MongoClient("localhost", 27017)
mongo_col = mongo_client['EssayEvalPro']['Essay']

# Get All Data From the DB
essays_list = [essay['essay'] for essay in mongo_col.find({},{"_id":0, 'essay':1})]

# Esaay Title Relevancy Model
model_title_essay = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Data Processing English language model in spaCy
nlp = spacy.load("en_core_web_sm") #exclude=['tagger','parser','ner','entity_linker']

# Grammar and spelling correction model
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
happy_tt_args = TTSettings(num_beams=3, min_length=1)


app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_file():

    if 'file' not in request.files:
        return 'Error: No file uploaded.'

    file = request.files['file']
    print("type of the file is ",type(file))    
    # Check if the file is empty
    if file.filename == '':
        return 'Error: Empty file uploaded.'
    
    # Get the optional parameters
    plag_thresh = float(request.form.get('plag_thresh', default=0.5))
    grammar_thresh = float(request.form.get('grammar_thresh', default=0.2))
    title = request.form.get('title', default=None)
    # print("parameters \n",plag_thresh,grammar_thresh,title)

    try:
        # Read the contents of the file
        file_content = file.read().decode('utf-8')
        # print("type(file_content) ",type(file_content))
        # print(file_content)

        # Extracting title from text file 
        if title is None:
            # Spliting Title and essay
            split_content = file_content.split('\n\n\n')
            title = split_content[0]; essay = split_content[1]
            # print('title ',title)
            # print('essay ',essay)
        
        else:
            essay = file_content

        #Grammar and Spelling Check
        grammar_result = Grammar_Spell_Check(text_process_pipeline=nlp, gs_model=happy_tt, gs_model_args=happy_tt_args, cosine_obj=cosine_1g, essay=essay)
        print("grammar_result ",grammar_result)

        # Preprocessing for plagiarism calaculcation
        preprocessed_essay = Preprocess(nlp,essay)
        
        # Plag result high result means low plag
        if len(essays_list) != 0:
            plag_result = plag_calculate(cosine_1g, cosine_2g, cosine_3g, essay, db_essay_list=essays_list)
        else:
            plag_result = 1
        
        print("plag result ",plag_result)

        # Calculating the relevance of the title with essay
        te_relevancy = Title_Essay_Relevancy(model=model_title_essay, essay=essay, title=title)
        print("Relevancy ",te_relevancy)

        # populating essay to mongo
        mongo_col.insert_one({'essay':preprocessed_essay})
        essays_list.append(preprocessed_essay)


        return str(plag_result)
    
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)

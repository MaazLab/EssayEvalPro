from flask import Flask, request
from sentence_transformers import SentenceTransformer
import spacy
import pymongo
from utils import Preprocess, Title_Essay_Relevancy,plag_calculate
from strsimpy import Cosine

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


app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_file():
    print("request.files ",request.files)
    # Check if the request contains a file
    if 'file' not in request.files:
        return 'Error: No file uploaded.'

    file = request.files['file']
    print("type of the file is ",type(file))    
    # Check if the file is empty
    if file.filename == '':
        return 'Error: Empty file uploaded.'

    try:
        # Read the contents of the file
        file_contents = file.read().decode('utf-8')
        print("type(file_contents) ",type(file_contents))
        print(file_contents)

        # Spliting Title and essay
        split_content = file_contents.split('\r\n\r\n\r\n')
        title = split_content[0]; essay = split_content[1]
        print('title ',title)
        print('essay ',essay)

        # Applying Preprocessing for plagiarism calaculcation
        preprocessed_essay = Preprocess(nlp,essay)
        
        if len(essays_list) == 0:
            print('No Plag Found')
        else:
            plag_result = plag_calculate(cosine_1g, cosine_2g, cosine_3g, essay, db_essay_list=essays_list)
        # Calculating the relevance of the title with essay
        te_relevancy = Title_Essay_Relevancy(model=model_title_essay, essay=essay, title=title)
        print(te_relevancy)

        # populating essay to mongo
        mongo_col.insert_one({'essay':preprocessed_essay})
        essays_list.append(preprocessed_essay)


        return str(plag_result)
    
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)

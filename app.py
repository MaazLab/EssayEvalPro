from flask import Flask, request
from sentence_transformers import SentenceTransformer
import pymongo
from utils import pdf_to_text, Preprocess, Title_Essay_Relevancy, plag_calculate, Grammar_Spell_Check
from happytransformer import HappyTextToText, TTSettings
from flask import Flask, jsonify
import tempfile

# MongoDB 
mongo_client = pymongo.MongoClient("localhost", 27017)
mongo_col = mongo_client['EssayEvalPro']['Essay']

# Get All Data From the DB
existing_essays = [essay['essay'] for essay in mongo_col.find({},{"_id":0, 'essay':1})]

# Esaay Title Relevancy Model
model_title_essay = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')



# Grammar and spelling correction model
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
happy_tt_args = TTSettings(num_beams=3, min_length=1)


app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_file():

    if 'file' not in request.files:
        return 'Error: No file uploaded.'

    file = request.files['file']    
    # Check if the file is empty
    if file.filename == '':
        return 'Error: Empty file uploaded.'
    
    
    # Get the optional parameters
    plag_thresh = float(request.form.get('plag_thresh', default=0.7))
    grammar_thresh = float(request.form.get('grammar_thresh', default=0.0))
    rel_thresh = float(request.form.get('rel_thresh', default=0.0))
    title = request.form.get('title', default=None)


    try:

        if file.filename.endswith('.pdf'):
            # Save the file as a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            file.save(temp_file.name)
            file_content = pdf_to_text(temp_file.name)
            if title is None:
                title = file_content.split('\n')[0]
                essay = file_content[len(title):]
            else:
                essay = file_content

        else:
            # Read the contents of the file
            file_content = file.read().decode('utf-8')


            # Extracting title from text file 
            if title is None:
                # Spliting Title and essay
                split_content = file_content.split('\r\n\r\n\r\n')
                title = split_content[0]; essay = split_content[1]        
            else:
                essay = file_content

        print('TITLE \n',title)
        print('\t\t\t\t-----------------')
        print('ESSAY \n',essay)
        print('\t\t\t\t-----------------')

        #Grammar and Spelling Check
        grammar_result, corrected_essay = Grammar_Spell_Check( gs_model=happy_tt, gs_model_args=happy_tt_args, essay=essay, thresh=grammar_thresh)
        
        print('CORRECTED ESSAY \n',corrected_essay)
        print('\t\t\t\t-----------------')
        # preprocessing document for DB and plag
        preprocessed_essay = Preprocess(essay=corrected_essay)

        # Plag result high result means low plag
        if len(existing_essays) != 0:
            matched_essay, plag_score = plag_calculate(essay=preprocessed_essay, essay_list=existing_essays)
            print("MATCHED ESSAY \n",matched_essay)
            if plag_score >= plag_thresh:
                return "Plag Found"
        else:
            plag_score = 0

        # Calculating the relevance of the title with essay
        te_relevancy = Title_Essay_Relevancy(model=model_title_essay, essay=essay, title=title, thresh=rel_thresh)

        # populating essay to mongo
        mongo_col.insert_one({'essay':preprocessed_essay})
        existing_essays.append(preprocessed_essay)
        
        final_score = ((te_relevancy + grammar_result - plag_score ) / 3 )*100
        print(f'Grammar Score {grammar_result}\n Relevancy Result {te_relevancy}\n Plag Score {plag_score}')
        print('Final Score ',final_score)
        result = {'score' : final_score,
                  'grammar' : grammar_result,
                  'relevancy' : te_relevancy,
                  'plag' : plag_score}
        return jsonify(result)
    
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True, port=5000)

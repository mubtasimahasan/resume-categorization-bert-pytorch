
import sys
import os
import shutil
import pandas as pd
from PyPDF2 import PdfReader
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import logging
logging.set_verbosity_error()

catogory_name = {0: 'HR', 1: 'DESIGNER', 2: 'INFORMATION-TECHNOLOGY', 3: 'TEACHER', 
                 4: 'ADVOCATE', 5: 'BUSINESS-DEVELOPMENT', 6: 'HEALTHCARE', 7: 'FITNESS', 
                 8: 'AGRICULTURE', 9: 'BPO', 10: 'SALES', 11: 'CONSULTANT', 12: 'DIGITAL-MEDIA', 
                 13: 'AUTOMOBILE', 14: 'CHEF', 15: 'FINANCE', 16: 'APPAREL', 17: 'ENGINEERING', 
                 18: 'ACCOUNTANT', 19: 'CONSTRUCTION', 20: 'PUBLIC-RELATIONS', 21: 'BANKING', 
                 22: 'ARTS', 23: 'AVIATION'}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True);

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(catogory_name),
                                                      output_attentions=False,
                                                      output_hidden_states=False).to(device);

model.load_state_dict(torch.load('./finetuned_BERT.model', map_location=torch.device(device)))
   

    
# category prediction function
def categorize(text):
    
    # tokenize the resume's text
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding='max_length', 
        truncation=True,
        max_length=256, 
        return_tensors='pt'
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    # predict with model
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.argmax(output[0], dim=1).cpu().numpy()
    
    return catogory_name[prediction[0]]



# functions to move files to predicted categories
def action(directory):

    
    file_names = []
    labels = []

    # iterate over the directory to get resumes' information
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file).replace('\\', '/')
                print('- Found resume location at',file_path)
                if os.path.isfile(file_path):
                    file_name = file_path.split('/')[-1].split('.')[0]
                    file_names.append(file_name)
                    pdf_reader = PdfReader(file_path)

            text = ''        
            for i in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[i]
                text += page.extract_text()
            
            cat_name = categorize(text)
            labels.append(cat_name)

            source_file = file_path
            destination_directory = './resume-categories-predicted/' + cat_name +'/'
            print(f'Copied "{file_name}" resume to the category folder destination "{destination_directory}"')

            os.makedirs(os.path.dirname(destination_directory), exist_ok=True)
            shutil.copy(source_file, destination_directory)

    csv_df = pd.DataFrame({'filename' : file_names, 'category': labels})
    csv_df.to_csv('./categorized_resumes.csv')
    print('> CSV file created at current location')

    

def main():
    args = sys.argv[1:]
    dir_path = args[0]
#     dir_path = './resume-dataset/data/test-data'
    action(dir_path)
    
    
if __name__ == '__main__':
    main()

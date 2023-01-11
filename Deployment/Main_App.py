# Delete all objects from memory

keys = list(globals().keys())

for o in keys:
    if not o.startswith('_'):
        print(o)
        del globals()[o]

# Imort from a file called Bert-medium.py

from Bert_medium import MediumBert
from Offensive_Bert import BertClassifier
from data_cleaning import cleaning_content
from Dialect_Bert import Dialect_Detection

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clear cache torch
torch.cuda.empty_cache()

print(torch.cuda.get_device_name(device))

from transformers import BertTokenizer, AutoTokenizer, BertTokenizerFast
import streamlit as st

# file path
import os

path_file = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path_file)

##########################FUNCTIONS########################

def predict_off(review_text,model,device,tokenizer):

        encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='longest',
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        #print(f'Review text: {review_text}')
        index = output.cpu().data.numpy().argmax()
        #print(f'Sentiment  : {index}')
        # decode the output of the model to get the predicted label
        pred = index
        
        return pred
#########################################""
def predict_other(review_text,model,device,tokenizer):
        
        encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=217,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='longest',
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        #print(f'Review text: {review_text}')
        index = output.cpu().data.numpy().argmax()
        #print(f'Sentiment  : {index}')
        # decode the output of the model to get the predicted label

        return index
#########################"##################

def predict_dialect(review_text,model,device,tokenizer):
        
        encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=123,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='longest',
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        #print(f'Review text: {review_text}')
        index = output.cpu().data.numpy().argmax()
        #print(f'Sentiment  : {index}')
        pred = index
        return pred


# Main prediction function

def predict(text,device,offensive_model,offensive_tokenizer,racism_model,misogyny_model,verbalabuse_model,dialect_model,religionhate_model,tokenizer_dialect,other_tokenizer,off_dictionary,racism_dict,misogyny_dict,verbalabuse_dict,dialect_dict,religionhate_dict):
        # clean text
        text = cleaning_content(text)
        
        # predict using offensive model
        off_pred = off_dictionary[predict_off(text,offensive_model,device,offensive_tokenizer)]

        if off_pred == 'offensive':
            # predict using racism model
            rac_pred = racism_dict[predict_other(text,racism_model,device,other_tokenizer)]
            # predict using misogyny model
            misog_pred = misogyny_dict[predict_other(text,misogyny_model,device,other_tokenizer)]
            # predict using verbal abuse model
            ver_pred = verbalabuse_dict[predict_other(text,verbalabuse_model,device,other_tokenizer)]
            # predict using dialect model
            dialect_pred = dialect_dict[predict_dialect(text,dialect_model,device,tokenizer_dialect)]
            # predict using religion hate model
            Religion_Hate_pred = religionhate_dict[predict_other(text,religionhate_model,device,other_tokenizer)]
            # return the prediction
            return {"Offensiveness": off_pred, "Dialect": dialect_pred, "Misogyny": misog_pred, "Racism": rac_pred, "Verbal Abuse": ver_pred, "Religion Hate": Religion_Hate_pred}
        
        # predict using misogyny model
        misog_pred = misogyny_dict[predict_other(text,misogyny_model,device,other_tokenizer)]
        # predict using dialect model
        dialect_pred = dialect_dict[predict_dialect(text,dialect_model,device,tokenizer_dialect)]
        
        # return the prediction  as a dataframe row
        return {"Offensiveness": off_pred, "Dialect": dialect_pred, "Misogyny": misog_pred, "Racism": "Not_Racism", "Verbal Abuse": "Not Verbal Abuse", "Religion Hate": "Not Religion Hate"}
###############################################

from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
import folium

geolocator = Nominatim(user_agent="NLP")

def geolocate(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan

# Stream lit app

st.title("Arabic Hate Speech Detection")

st.write("This app detects hate speech in Arabic tweets")

st.write("Please enter your text below")


# Session state
if 'Loaded' not in st.session_state:
    print('Loading model ysk')
    st.session_state['Loaded'] = False
else:
    print('Model already loaded')
    st.session_state['Loaded'] = True
    

if st.session_state['Loaded'] == False:

    # clear cache torch
    torch.cuda.empty_cache()
    # Offensiveness detection model 

    offensive_model = BertClassifier()
    offensive_model.load_state_dict(torch.load(os.path.join(parent_path,'models\modelv3.pt')))
    offensive_tokenizer = BertTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02', do_lower_case=True)

    #send model to device

    offensive_model = offensive_model.to(device)
    st.session_state['Offensive_model'] = offensive_model
    st.session_state['Offensive_tokenizer'] = offensive_tokenizer
    print('Offensive model loaded')
    off_dictionary = {1: 'offensive', 0: 'non_offensive'}
    st.session_state['Offensive_dictionary'] = off_dictionary

    ##############################################################################################################################

    # Other four models

    other_tokenizer =  AutoTokenizer.from_pretrained("asafaya/bert-medium-arabic")
    st.session_state['Other_tokenizer'] = other_tokenizer

    racism_model,religionhate_model,verbalabuse_model,misogyny_model = MediumBert(),MediumBert(),MediumBert(),MediumBert()
    ################################################################

    racism_model.load_state_dict(torch.load(os.path.join(parent_path,'models\\racism\\racism_arabert.pt')))
    racism_dict = {0: 'non_racist', 1: 'racist'}

    racism_model = racism_model.to(device)

    st.session_state['Racism_model'] = racism_model
    st.session_state['Racism_dictionary'] = racism_dict

    print('Racism model loaded')
    ################################################################

    religionhate_model.load_state_dict(torch.load(os.path.join(parent_path,'models\\religion_hate\\religion_hate_params.pt')))
    religionhate_dict = {0: 'Religion Hate', 1: 'Not Religion Hate'}

    religionhate_model = religionhate_model.to(device)

    st.session_state['Religion_hate_model'] = religionhate_model
    st.session_state['Religion_hate_dictionary'] = religionhate_dict

    print('Religion Hate model loaded')
    ################################################################

    verbalabuse_model.load_state_dict(torch.load(os.path.join(parent_path,'models\\verbal_abuse\\verbal_abuse_arabert.pt')))
    verbalabuse_dict = {0: 'Verbal Abuse', 1: 'Not Verbal Abuse'}

    verbalabuse_model=verbalabuse_model.to(device)

    st.session_state['Verbal_abuse_model'] = verbalabuse_model
    st.session_state['Verbal_abuse_dictionary'] = verbalabuse_dict

    print('Verbal Abuse model loaded')
    ################################################################

    misogyny_model.load_state_dict(torch.load(os.path.join(parent_path,'models\\misogyny\\misogyny.pt')))
    misogyny_dict = {0: 'misogyny', 1: 'non_misogyny'}

    misogyny_model=misogyny_model.to(device)

    st.session_state['Misogyny_model'] = misogyny_model
    st.session_state['Misogyny_dictionary'] = misogyny_dict


    print('Misogyny model loaded')
    ################################################################

    # Dialect detection model

    dialect_model = Dialect_Detection(10)
    dialect_model.load_state_dict(torch.load(os.path.join(parent_path,'models\\dialect_classifier.pt')))

    dialect_model = dialect_model.to(device)

    st.session_state['Dialect_model'] = dialect_model

    print('Dialect model loaded')

    tokenizer_dialect = BertTokenizerFast.from_pretrained('alger-ia/dziribert')

    st.session_state['Dialect_tokenizer'] = tokenizer_dialect

    # load the model
    dialect_dict = {0: 'lebanon', 1: 'egypt', 2: 'morocco', 3: 'tunisia', 4: 'algeria', 5: 'qatar', 6: 'iraq', 7: 'saudi arabia', 8: 'libya', 9: 'jordan'}

    st.session_state['Dialect_dictionary'] = dialect_dict

    st.session_state['Loaded'] = True

text = st.text_area("Enter Text", "Type Here")

if st.button("Predict"):
    result = predict(text = text, device = device,
                    offensive_model= st.session_state['Offensive_model'],
                    offensive_tokenizer= st.session_state['Offensive_tokenizer'],
                    racism_model= st.session_state['Racism_model'],
                    misogyny_model=st.session_state['Misogyny_model'],
                    verbalabuse_model= st.session_state['Verbal_abuse_model'],
                    dialect_model=st.session_state['Dialect_model'],
                    religionhate_model=st.session_state['Religion_hate_model'],
                    tokenizer_dialect=st.session_state['Dialect_tokenizer'],
                    other_tokenizer=st.session_state['Other_tokenizer'],
                    off_dictionary=st.session_state['Offensive_dictionary'],
                    racism_dict=st.session_state['Racism_dictionary'],
                    misogyny_dict=st.session_state['Misogyny_dictionary'],
                    verbalabuse_dict=st.session_state['Verbal_abuse_dictionary'],
                    dialect_dict=st.session_state['Dialect_dictionary'],
                    religionhate_dict=st.session_state['Religion_hate_dictionary'])

    st.write(result)

    location  = geolocate(result['Dialect'])

    # map with contry highlited
    location = pd.DataFrame({'lat': [location[0]], 'lon': [location[1]]})
    st.map(data= location , zoom=5)

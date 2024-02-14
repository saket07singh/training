#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_lg  #importing the large model for english')


# In[13]:


import spacy


spacy_lang = spacy.load("en_core_web_lg")


# In[14]:


spacy_lang


# In[48]:


#create a document object

doc= spacy_lang("Ravi Jackson is appointed as the new CEO of Albertco. He lives in Washington")


# In[49]:


doc


# In[50]:


type(doc)


# In[51]:


doc.ents


# In[19]:


type(doc.ents)


# In[20]:


type(doc.ents[0])


# In[21]:


#importing displacy which is a visualisation model for predicted entities

from spacy import displacy

#style='ent' to visualise the entities

displacy.render(doc, style='ent', jupyter=True)


# In[53]:


doc2= spacy_lang("Antiretroviral therapy (ART) is recommended for all HIV-infectedindividuals to reduce the risk of disease progression.\nART also is recommended for HIV-infected individuals for the prevention of transmission of HIV.\nPatients starting ART should be willing and able to commit to treatment and understand thebenefits and risks of therapy and the importance of adherence. Patients may chooseto postpone therapy, and providers, on a case-by-case basis, may elect to defertherapy on the basis of clinical and/or psychosocial factors.")

displacy.render(doc2, style='ent', jupyter=True)


# In[54]:


doc3 = spacy_lang("While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.")


spacy.displacy.render(doc3, style="ent", jupyter=True)


# In[22]:


#TRAINING SPECIFIC TO MEDICAL DATA

import json

with open("Corona2.json", 'r') as data:
    data=json.load(data)
data#["examples"]


# In[23]:


data["examples"]


# In[24]:


#each exmaple is a dictionary

data["examples"][0].keys()


# In[25]:


data["examples"]


# In[26]:


data["examples"][0]["content"]


# In[27]:


data["examples"][0]["annotations"][1]


# In[28]:


#every entity in the "content" of an example has been annotated in "annotations"


# training data- "content", 'start' and 'end' of entity in conent and  and the the "tag_name"

train_set=[]
for example in data["examples"]:
    dict={}
    dict["text"]=example["content"]
    dict["entities"]=[]
    for annot in example["annotations"]:
        start=annot['start']
        end=annot['end']
        ent_label=annot['tag_name'].lower()
        dict["entities"].append((start,end,ent_label))
    train_set.append(dict)
    
train_set[2]   #list of dictionaries


# In[29]:


train_set[0]["text"][211:214]


# In[ ]:


#model training


# In[31]:


#data to be in Doc bin format

from spacy.tokens import DocBin
from tqdm import tqdm
nlp = spacy.blank("en")              # load a new blank spacy model
doc_bin = DocBin()
     


# In[32]:


from spacy.util import filter_spans

for training  in tqdm(train_set): 
   
    text = training.get('text', '') 
    entities = training.get('entities', [])
    doc = nlp.make_doc(text) 
    spans = []
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            spans.append(span)
    filtered_ents = filter_spans(spans)
    doc.ents = filtered_ents 
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy") 


# In[ ]:


#https://spacy.io/usage/training#quickstart    -- for the base config file


# In[33]:


get_ipython().system('python -m spacy init fill-config base_config.cfg config.cfg')
     


# In[34]:


import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# In[40]:



get_ipython().system('python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./train.spacy ')
     


# In[41]:


nlp_ner = spacy.load("model-best")


# In[45]:


doc = nlp_ner("While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.")


spacy.displacy.render(doc, style="ent", jupyter=True)


# In[44]:


doc2 = nlp_ner("Antiretroviral therapy (ART) is recommended for all HIV-infectedindividuals to reduce the risk of disease progression.\nART also is recommended for HIV-infected individuals for the prevention of transmission of HIV.\nPatients starting ART should be willing and able to commit to treatment and understand thebenefits and risks of therapy and the importance of adherence. Patients may chooseto postpone therapy, and providers, on a case-by-case basis, may elect to defertherapy on the basis of clinical and/or psychosocial factors.")


spacy.displacy.render(doc2, style="ent", jupyter=True)


# In[ ]:





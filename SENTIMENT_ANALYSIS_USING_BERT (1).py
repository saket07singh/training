#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt


# In[35]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)
get_ipython().system('pip install peft')
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
get_ipython().system('pip install evaluate')
import evaluate
import torch
from transformers import TFAutoModelForSequenceClassification


# In[36]:


#load dataset



dataframe= pd.read_csv("Tweets.csv",delimiter=',', encoding="utf-8-sig")


# In[37]:


dataframe.head(10)


# In[38]:


#extract text and airline_sentiment from above

dataframe= dataframe[["airline_sentiment", "text"]]
dataframe


# In[39]:


dataframe = dataframe.dropna(subset=['text', 'airline_sentiment'])
print("NaN values in the dataset:")


# In[40]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataframe['Sentiment'] = label_encoder.fit_transform(dataframe['airline_sentiment'])


# In[41]:


dataframe


# In[42]:


dataframe = dataframe.drop('airline_sentiment', axis=1)


# In[43]:


dataframe


# In[44]:


dataframe = dataframe.sample(frac=0.3, random_state=42).reset_index(drop=True)


# In[12]:


dataframe


# In[14]:


dataframe


# In[15]:


import re
dataframe.columns = dataframe.columns.str.strip()
# Handle missing values
#dataframe = dataframe.dropna()  # Drop rows with any NaN values

# Remove special characters from the text column
dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))

# Remove duplicate rows
dataframe = dataframe.drop_duplicates()
dataframe


# In[16]:


#checking the sentiment column for unique values

print(dataframe.Sentiment.unique())


# In[17]:


unique_sentiments = dataframe['Sentiment'].unique()
counter = []

for sentiment in unique_sentiments:
    count = dataframe['Sentiment'].value_counts()[sentiment]
    counter.append(count)

counter


# In[18]:


train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)


# In[19]:


train_df


# In[20]:


test_df


# In[21]:


len(label_encoder.classes_)


# In[22]:


model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))


# In[23]:


train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=256)


# In[24]:


train_encodings


# In[26]:


data_collator = DataCollatorWithPadding(tokenizer)
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


# In[27]:


train_dataset = CustomDataset(train_encodings, train_df['Sentiment'].tolist())
test_dataset = CustomDataset(test_encodings, test_df['Sentiment'].tolist())


# In[28]:


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[29]:


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

def evaluation_metrics(pred):
    labels= pred.label_ids
    preds=pred.predictions.argmax(-1)
    precision, recall, f1, _= precision_recall_fscore_support(labels, preds, average="weighted", pos_label=0)
    cm=confusion_matrix(labels, preds, labels=[0,1,2])
    
    accuracy=accuracy_score(labels, preds)
    return {
        "y_true": labels,
        "y_pred": preds,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix":cm
        
    }


# In[30]:


from huggingface_hub import notebook_login
notebook_login()


# In[31]:


from transformers import TrainingArguments, Trainer
 
repo_name = "finetuning-sentiment-model"
 
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=train_dataset,
   eval_dataset=test_dataset,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=evaluation_metrics,
)


# In[32]:


trainer.train()


# In[33]:


trainer.evaluate()


# In[ ]:





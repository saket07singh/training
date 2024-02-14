#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


# In[2]:


data= pd.read_csv("Language_Detection.csv")


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


#CHECK NO OF LANGUAGES

data['Language'].value_counts()


# In[59]:


plt.figure(figsize=(10,10))
sb.countplot(data['Language'])


# In[7]:


#REMOVE STOPWORDS AND CHARACTERS FROM THE TEXTS

PS= PorterStemmer()
corpus=[]

for i in range(len(data['Text'])):
    new= re.sub('^[a-zA-Z]', ' ', data['Text'][i])
    new=new.lower()
    new=new.split()
    new = [PS.stem(word) for word in new if word not in stopwords.words()]
    new=' '.join(new)
    corpus.append(new)
    
    print(f"{i}")


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=10000)
x= cv.fit_transform(corpus).toarray()


# In[9]:


x.shape

corpus
# In[11]:


#encode all language labels

from sklearn.preprocessing import LabelEncoder
labels=LabelEncoder()
y= labels.fit_transform(data['Language'])


# In[12]:


y


# In[13]:


len(y)


# In[14]:


#create a final data frame

dataframe= pd.DataFrame(np.c_[corpus,y], columns=['Sentences','Languages'])


# In[15]:


dataframe


# In[16]:


#split training and testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)


# In[18]:


x_train.shape


# In[19]:


x_test.shape


# In[20]:


y_train.shape


# In[21]:


y_test.shape


# In[23]:


#model

from sklearn.naive_bayes import MultinomialNB


# In[25]:


model=MultinomialNB().fit(x_train,y_train)


# In[26]:


prediction=model.predict(x_test)


# In[28]:


prediction


# In[29]:


y_test


# In[31]:


#evaluation

from sklearn.metrics import accuracy_score, confusion_matrix


# In[33]:


print(accuracy_score(y_test,prediction))
print(confusion_matrix(y_test,prediction))


# In[60]:


#heatmap

plt.figure(figsize=(10,10))
sb.heatmap(confusion_matrix(y_test,prediction), annot=True, cmap=plt.cm.Accent)


# In[36]:


#actual and predicted data

outcomes=pd.DataFrame(np.c_[y_test,prediction], columns=['Actual',"Predcicted"])
outcomes


# In[46]:


# Mapping between original labels and encoded labels
label_mapping = dict(zip(labels.classes_, labels.transform(labels.classes_)))
label_mapping


# In[54]:


#testing the model build a function


def language_classifier(sentence):
    languages={'Arabic': 0,
       'Danish': 1,
        'Dutch': 2,
       'English': 3,
        'French': 4,
        'German': 5,
        'Greek': 6,
        'Hindi': 7,
      'Italian': 8,
       'Kannada': 9,
     'Malayalam': 10,
     'Portugeese': 11,
       'Russian': 12,
       'Spanish': 13,
       'Sweedish': 14,
        'Tamil': 15,
      'Turkish': 16}
    
    new= re.sub('^[a-zA-Z]', ' ', sentence)
    new=new.lower()
    new=new.split()
    new = [PS.stem(word) for word in new if word not in stopwords.words()]
    new=' '.join(new)
    
    new= cv.transform([new]).toarray()
    output=model.predict(new)[0]
    
    keys=list(languages.keys())
    values=list(languages.values())
    
    position=values.index(output)
    
    output=keys[position]
    
    print(output)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    


# In[55]:


language_classifier("my name is saket")


# In[56]:


language_classifier("ನನ್ನ ಹೆಸರು ಸಾಕೇತ್")


# In[ ]:





# In[ ]:





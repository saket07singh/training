#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt


# In[66]:


#loading the dataset as a pandas dataframe and specifying the names of the columns 

dataframe= pd.read_csv("Tweets.csv")


# In[20]:


dataframe.head(5)


# In[23]:


#extract text and airline_sentiment from above

dataframe= dataframe[["airline_sentiment", "text"]]
dataframe


# In[24]:


dataframe.info()


# In[25]:


dataframe.shape


# In[26]:


#checking the sentiment column for unique values

print(dataframe.airline_sentiment.unique())


# In[77]:


# visualizing all unique sentiments

unique_sentiments= dataframe.airline_sentiment.unique()
counter=[]
for i in uniquesentiments:
    count= len(dataframe[dataframe.airline_sentiment==i])
    counter.append(count)
counter


# In[78]:


plt.bar(['neutral', 'positive','negative'], counter)
plt.show


# In[36]:


# Encode sentiment labels to numerical values
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
dataframe['encoded_sentiment'] = dataframe['airline_sentiment'].map(sentiment_mapping)
dataframe.head()


# In[37]:


#split dataset into testing and training
x=dataframe["text"]
y=dataframe["encoded_sentiment"]


# In[38]:


x


# In[39]:


y


# In[41]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)


# In[42]:


#vectorizing the text using count vectorizer 

vectorizer= CountVectorizer()
vectorizer.fit(x_train)
x_train_vec=vectorizer.transform(x_train)
x_test_vec=vectorizer.transform(x_test)


# In[43]:


x_train[0]


# In[45]:


print(x_train_vec[0])


# In[49]:


# Train a logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(x_train_vec, y_train)

# Predict sentiment on the test set
y_pred = model.predict(x_test_vec)


# In[51]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[54]:


#decode numerical labels to original
decoded_predictions = pd.Series(y_pred).map({0: 'negative', 1: 'neutral', 2: 'positive'})


# In[63]:



# pi chart of predictions
unique_values, counts = np.unique(decoded_predictions, return_counts=True)

labels = unique_values
sizes = counts

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'blue', 'green'])
plt.title('Sentiment Distribution in Test Set')
plt.show()


# In[79]:



from sklearn.metrics import confusion_matrix

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sb.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds', xticklabels=sentiment_mapping.keys(), yticklabels=sentiment_mapping.keys())
plt.xlabel('Predicted Test Result')
plt.ylabel('True Test Result')
plt.title('Confusion Matrix for true vs predicted on testing data')
plt.show()


# In[70]:


conf_mat


#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[38]:


# Loading the dataset
df = pd.read_csv('WELFake_Dataset.csv')


df.head()


# In[39]:


# Count the occurrences of each label 1 for real and 0 for fake
label_counts = df['label'].value_counts()

# Create a pie chart
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
plt.title('Fake vs Real news Distribution')
plt.show()


# In[40]:


df = df.dropna()


# In[41]:


X = df['text']
y = df['label']


# In[42]:


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# In[43]:


print(y_encoded)


# In[45]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# In[48]:


X_train


# In[49]:


# Tokenize and pad sequences
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)


# In[52]:


len(X_train)


# In[53]:


X_train_seq = tokenizer.texts_to_sequences(X_train)   #each news is tokenized
X_test_seq = tokenizer.texts_to_sequences(X_test)


# In[54]:


len(X_train_seq)


# In[55]:


X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')


# In[56]:


# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))


# In[57]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[58]:


# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)


# In[59]:


# Evaluate the model
y_pred_proba = model.predict(X_test_pad)
y_pred = np.round(y_pred_proba).astype(int)


# In[60]:


r2 = r2_score(y_test, y_pred_proba)
r2


# In[61]:


# Display classification report
print(classification_report(y_test, y_pred))


# In[62]:


# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[69]:


# Plot heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# Create a single subplot
fig, ax = plt.subplots(figsize=(12, 5))

# Heatmap for Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'], ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix Heatmap')
plt.show()


# In[ ]:





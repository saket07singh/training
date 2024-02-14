#!/usr/bin/env python
# coding: utf-8

# In[5]:


from nltk.tokenize import word_tokenize, sent_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer


# In[27]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class TextSummarizer_tfidf:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words("english"))
        self.freqTable = dict()
        self.average = 0

    def tokenize_text(self, text):
        words = word_tokenize(text.lower())
        return [word for word in words if word.isalnum() and word not in self.stop_words]

    def create_frequency_table(self, words):
        for word in words:
            if word in self.freqTable:
                self.freqTable[word] += 1
            else:
                self.freqTable[word] = 1

    def score_sentences(self, sentences):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)

        sentence_value = dict()

        for i, sentence in enumerate(sentences):
            value = 0
            for word, freq in self.freqTable.items():
                if word in vectorizer.vocabulary_:
                    tf_idf = X[i, vectorizer.vocabulary_[word]]
                    value += tf_idf

            sentence_value[sentence] = value

        return sentence_value

    def generate_summary(self, text):
        words = self.tokenize_text(text)
        self.create_frequency_table(words)

        sentences = sent_tokenize(text)
        sentence_value = self.score_sentences(sentences)

        sum_values = sum(sentence_value.values())
        self.average = int(sum_values / len(sentence_value))

        summary = " ".join(sentence for sentence in sentences if
                            sentence in sentence_value and sentence_value[sentence] > (1.2 * self.average))

        return summary


# In[28]:


# Import the TextSummarizer class from your module
#from Text_summarizer import TextSummarizer

# Create an instance of the TextSummarizer
summarizer = TextSummarizer_tfidf()

# Define your text for summarization
text = """
In common usage, climate change describes global warming—the ongoing increase in global average temperature—and its effects on Earth's climate system. Climate change in a broader sense also includes previous long-term changes to Earth's climate. The current rise in global average temperature is more rapid than previous changes, and is primarily caused by humans burning fossil fuels
"""
# Generate the summary
summary = summarizer.generate_summary(text)

# Print the generated summary
print("Generated Summary:")
print(summary)


# In[ ]:





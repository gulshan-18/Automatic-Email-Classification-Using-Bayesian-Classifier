# https://www.kaggle.com/harshsinha1234/email-spam-classification-nlp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('emails.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('emails.csv')
df.head()


df.shape

df.info()

df['spam'].value_counts()

#sns.countplot(df['spam'])

from nltk import word_tokenize

def count_words(text):
    words = word_tokenize(text)
    return len(words)

import nltk
nltk.download('stopwords')
nltk.download('punkt')

df['count']=df['text'].apply(count_words)

df['count']

df.groupby('spam')['count'].mean()

import string
from nltk.corpus import stopwords

def process_text(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    
    
    return ' '.join([word for word in no_punc.split() if word.lower() not in stopwords.words('english')])


df['text']=df['text'].apply(process_text)

df['text']

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def stemming (text):
    return ''.join([stemmer.stem(word) for word in text])

df['text']=df['text'].apply(stemming)

df.head()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer= CountVectorizer()
message_bow = vectorizer.fit_transform(df['text'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(message_bow,df['spam'],test_size=0.20)

from sklearn.naive_bayes import MultinomialNB
nb= MultinomialNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#from sklearn.metrics import plot_confusion_matrix
#plot_confusion_matrix(nb,X_test,y_test)
#import seaborn as sns
#from sklearn.metrics import confusion_matrix as cm
#cm1 = cm(y_test, y_pred['naive_bayes'])
#print(cm1)
#sns.heatmap(cm1,annot=True,annot_kws={'size':14}, fmt='d').set_title('Confusion Matrix')

filename='finalized_model.sav'
joblib.dump(nb, filename)

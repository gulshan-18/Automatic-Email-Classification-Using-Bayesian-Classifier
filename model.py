import numpy as np 
import pandas as pd 
import warnings
import joblib
warnings.filterwarnings('ignore')

# importing the dataset
dataset = pd.read_csv('emails.csv', encoding='latin-1')
dataset.head()

# for splitting dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset["text"],dataset["spam"], test_size = 0.2, random_state = 10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# for vectorizing words
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(stop_words='english')
vect.fit(X_train)

print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)
type(X_test_df)

prediction = dict()
# Naive Bayes Machine Learning Model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df,y_train)

prediction["naive_bayes"] = model.predict(X_test_df)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# get accuracy
accuracy_score(y_test,prediction["naive_bayes"])

print(classification_report(y_test, prediction['naive_bayes'], target_names = ["Ham", "Spam"]))

filename='finalized_model.sav'
joblib.dump(model, filename)
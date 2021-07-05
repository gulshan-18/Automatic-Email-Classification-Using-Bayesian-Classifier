import numpy as np 
import pandas as pd 
from django.http import HttpResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import joblib

def index(request):
    return render(request,"index.html")

def home(request):
    return render(request,"home.html")

def result(request):

    model=joblib.load("finalized_model.sav")
    #lis=[]
    #lis.append(request.GET['text'])
    #ans=model.predict([lis])
    #return render(request,"result.html",{'ans':ans})
    dataset = pd.read_csv('D:\Model\emails.csv', encoding='latin-1')
    # for splitting dataset into train set and test set
    X_train,X_test,y_train,y_test = train_test_split(dataset["text"],dataset["spam"], test_size = 0.2, random_state = 10)
    vect = CountVectorizer(stop_words='english')
    S="Subject: save your money buy getting this thing here  you have not tried cialls yet ?  than you cannot even imagine what it is like to be a real man in bed !  the thing is that a great errrectlon is provided for you exactiy when you want .  cialis has a lot of advantages over viagra  - the effect lasts 36 hours !  - you are ready to start within just 10 minutes !  - you can mix it with aicohoi ! we ship to any country !  get it right now ! . "
    S=[S]
    vect.fit(X_train) 
    #S=request.GET['text']
    # print(S)
    # S=[S]
    print(S)
    S_df = vect.transform(S)
    type(S_df)
    ans=model.predict(S_df)
    print(ans)
    return render(request,"result.html",{'ans':ans})

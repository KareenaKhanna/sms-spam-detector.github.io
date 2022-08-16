# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:46:46 2022

@author: User
"""

import streamlit as st
import string
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_txt(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    l=[]
    for i in text:
        if i.isalnum():
            l.append(i)
    text=l[:]
    l.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)
    
    text=l[:]
    l.clear()
    
    for i in text:
        l.append(ps.stem(i))
            
    return " ".join(l)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    

    #preprocess
    transformed_sms = transform_txt(input_sms)
    #vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display
    if result==1:
        st.header("Its a Spam")
    else:
        st.header("Its not a Spam")

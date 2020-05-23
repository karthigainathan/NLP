# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:55:25 2020

@author: karth
"""


#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages


#Reading the text file
with open("C:\\Users\\karth\Desktop\\covid.txt",'r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#Tokenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings","what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating Bot response
def response(user_response):
    CoronaBoT_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
       CoronaBoT_response=CoronaBoT_response+"I am sorry! I don't understand you"
       return CoronaBoT_response
    else:
        CoronaBoT_response = CoronaBoT_response+sent_tokens[idx]
        return CoronaBoT_response


flag=True
print("CoronaBoT: Hi,  I am here to answer your queries about Covid-19. If you want to exit, Please type Bye!")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' or user_response=='thanks alot'):
            flag=False
            print("CoronaBoT: You are welcome...")
        else:
            if(greeting(user_response)!=None):
                print("CoronaBoT: "+greeting(user_response))
            else:
                print("CoronaBoT: ",end="")
                print(response(user_response)) 
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("CoronaBoT: Bye! take care...")
        
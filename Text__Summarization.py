# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:52:38 2020

@author: karth
"""
#Lexrank

from sumy.parsers. plaintext import Plaintextparser from
sumy.nlp.tokenizers import tokenizer
from sumy. Summarizer.lex_rank import LexRankSummarizer
file = “a19.txt”
parser = Plaintextparser.from_file(fie,Tokenizer(“English”))
summarizer = LexRankSummarizer()
summary = summarizer(parse.document, 5)
lexcontent=’’
for sentence in summary:
       lexcontent=lexcontent+str(sentence)
print(“printing lex xontent”)
print(lexcontent)


------------------------------------------------------------------------
#LSA

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
LANGUAGE = "english"
SENTENCES_COUNT = 10
 
if __name__ == "__main__":
    parser = PlaintextParser.from_file("a19.txt", Tokenizer(LANGUAGE))   
    print ("--LsaSummarizer--")    
    lsasummarizer = LsaSummarizer()
    lsasummarizer = LsaSummarizer(Stemmer(LANGUAGE))
    lsasummarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in lsasummarizer(parser.document, SENTENCES_COUNT):
        lsacontent=lsacontent+str(sentence)
        print(sentence)
        
------------------------------------------------------------------------------
#LDA

from gensim.summarization import summarize
from gensim.summarization import keywords
from rouge import Rouge 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

'''Read file into Python'''
f = open('ten issues.txt')
file = "ten issues.txt"

message = f.read()
stopWords = set(stopwords.words("english"))
words = word_tokenize(message)

'''Summarizing using Custom method'''
freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1
sentences = sent_tokenize(message)


sentenceValue = dict()
for sentence in sentences:
    for wordValue in freqTable:
        if wordValue in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freqTable[wordValue]
            else:
                sentenceValue[sentence] = freqTable[wordValue]
sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

# Average value of a sentence from original text
average = int(sumValues/ len(sentenceValue))
customsummary = ''
for sentence in sentences:
        if sentence in sentenceValue and sentenceValue[sentence] > (1.5 * average):
            customsummary +=  " " + sentence
print ("--CUSTOM Summarizer--") 
print(customsummary)

with open("customsummary.txt", "w") as text_file:
    text_file.write(customsummary)

'''Summarizing using LDA'''
print ("--LDaSummarizer--") 
print (summarize(message, ratio=0.01))
ldasummary=summarize(message, ratio=0.01)

with open("ldasummary.txt", "w") as text_file:
    text_file.write(ldasummary)

rouge = Rouge()
scores = rouge.get_scores(customsummary, ldasummary)
print("Printing Rouge scores")
print(scores)


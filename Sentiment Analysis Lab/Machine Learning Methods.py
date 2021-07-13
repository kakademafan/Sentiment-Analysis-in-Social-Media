#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:26:08 2019

@author: lucas
"""


'''
The dataset is from the weblink  
    https://www.kaggle.com/kazanova/sentiment140/version/2
'''

import pandas as pd

df = pd.read_csv("./sentiment140/sentiment140_sample.csv")
df = df.iloc[:,[0,5]]
df.columns = ['sentiment','text']
df.loc[df['sentiment'] == 4 ,'sentiment'] = 'positive'
df.loc[df['sentiment'] == 0 ,'sentiment'] = 'negative'

# shuffle rows of dataframe
df = df.sample(frac = 1, random_state = 100)

df




### Preprocess Text ###
from time import time
from basic_functions import *

import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer 

stoplist = stopwords.words('english')
lemmatizer = WordNetLemmatizer() # set lemmatizer
stemmer = PorterStemmer() # set stemmer
processed_text = [] #used to store the preprocessed text and sentiment


def process_word(text, textID, y):
    
    # tokens of one sentence each time
    SentenceTokens = [] 

    # it takes a text as an input and provides a list of every token in it
    #e.g. text = "To be honest, I am OK.",  nltk.word_tokenize(text) =  ['To', 'be', 'honest', ',', 'I', 'am', 'OK', '.']    
    tokens = nltk.word_tokenize(text) 
    
    for w in tokens:

        if  (w not in stoplist):
            final_word = w.lower()
            #e.g. replaceElongated("laaaaaugh") = "laugh"
            final_word = replaceElongated(final_word)
            #e.g. lemmatizer.lemmatize("cats") = "cat"  lemmatizer.lemmatize("plays") = "play"
            final_word = lemmatizer.lemmatize(final_word)
            #e.g. stemmer.stem("cat's") = "cat"  stemmer.stem("joyful") = "joy"
            final_word = stemmer.stem(final_word) 
            SentenceTokens.append(final_word)           
         
    Sentence = " ".join(SentenceTokens) # form again the sentence from the list of tokens
    processed_text.append(str(textID)+"\t"+ str(y)+"\t"+Sentence)
    


def pre_process(text,textID, y):
    text = removeUnicode(text) 
    text = removeURL(text) 
    text = removeAtUser(text) 
    text = removeHashtagInFrontOfWord(text) 

    text = replaceContraction(text) # replaces contractions to their equivalents
    text = removeNumbers(text) # remove integers from text

    text = replaceEmoticons(text)
    text = removeEmoticons(text) # removes emoticons from text
    text = replaceMultiExclamationMark(text) # replaces repetitions of exlamation marks with the tag "multiExclamation"
    text = replaceMultiQuestionMark(text) # replaces repetitions of question marks with the tag "multiQuestion"
    text = replaceMultiStopMark(text) # replaces repetitions of stop marks with the tag "multiStop" 
    text = removePunctuation(text)
    
    process_word(text, textID, y)  



from tqdm import tqdm 

# Get the list of processed_text
t0 = time()

for textID in tqdm(df.index): #tqdm is used to monitor process, you can delete it for simplicity      
    y = df.loc[textID,'sentiment'] 
    text = df.loc[textID,'text']     
    pre_process(text, textID, y)
    
t1 = time()
print('\n Total processing time', round(t1-t0, 2),'s')

processed_text

# Convert the processed_text from list to dataframe
import pandas as pd     
processed_text = [line.split('\t') for line in processed_text]
processed_df = pd.DataFrame(processed_text)
processed_df.rename(columns={0:'ID',1:'sentiment',2:'text'},inplace=True)

processed_df


### Machine Learning Methods ### 

# Construct training set and test set
train_ML_x = processed_df['text'][:int(0.8*len(processed_df))]
train_ML_y = processed_df['sentiment'][:int(0.8*len(processed_df))] 
    
test_ML_x = processed_df['text'][int(0.8*len(processed_df)):]
test_ML_y = processed_df['sentiment'][int(0.8*len(processed_df)):] 


train_ML_x
train_ML_y

# look at the item frequency in test set
from scipy import stats
stats.itemfreq(test_ML_y)



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


### SVM with count as value of features ###
unigram_bigram_clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer="word",
                                   ngram_range=(1, 2)# use unigram and bigram as features
                                   )),
    ('classifier', LinearSVC()) # LinearSVC means support vector machine classifier with linear kernel
])
unigram_bigram_clf.fit(train_ML_x, train_ML_y)
unigram_bigram_clf.predict(test_ML_x)


test_ML_pred_y = unigram_bigram_clf.predict(test_ML_x)
test_ML_pred_y


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(test_ML_y, test_ML_pred_y)
accuracy_score(test_ML_y, test_ML_pred_y)


'''
Q1: Predict with TFIDF as feature values 
'''




'''
Q2: Predict with Airline dataset, which is a 3-class dataset
'''



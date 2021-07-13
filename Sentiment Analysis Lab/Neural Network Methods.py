#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:35:07 2019

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





### Get word representation with Word2Vec  ###
import keras.backend as K
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec

# Create Word2Vec for tokens in the context of current processed text
vector_size = 300
tokenized_text = [nltk.word_tokenize(u) for u in processed_df["text"]] 

word2vec = Word2Vec(sentences= tokenized_text,
                    size=vector_size, 
                    window=15, #The maximum distance between the current and predicted word
                    negative=20, # negative sampling ratio
                    iter=500, # number of iteration over the corpus
                    seed=1000,
                    workers=multiprocessing.cpu_count())

X_vecs = word2vec.wv
del word2vec

X_vecs


### Construct train and test sets ###

train_size = int(0.8*len(processed_df))
test_size = len(processed_df) - train_size
max_tweet_length = 30 # maximum number of tokens used in each sentence

'''
In 3-class classification, the following should be revised
'''
train_DL_x = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
train_DL_y = np.zeros((train_size, 2), dtype=np.int32)
test_DL_x = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
test_DL_y = np.zeros((test_size, 2), dtype=np.int32)

for i in range(len(processed_df)):
    for t, token in enumerate(tokenized_text[i]):
        if t >= max_tweet_length:
            break
        
        if token not in X_vecs:
           continue
    
        if i < train_size:
            train_DL_x[i, t, :] = X_vecs[token]
        else:
            test_DL_x[i - train_size, t, :] = X_vecs[token]
            
    if i < train_size:
        train_DL_y[i, :] = [1.0, 0.0] if processed_df['sentiment'][i] == "negative" else [0.0, 1.0]
    else:
        test_DL_y[i - train_size, :] = [1.0, 0.0] if processed_df['sentiment'][i] == "negative" else [0.0, 1.0]
  

### CNN model ###
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D 
from keras.optimizers import Adam


# Set random seed (for reproducibility)
np.random.seed(1000)

# Keras convolutional model
model = Sequential()

model.add(Conv1D(200, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(max_tweet_length, vector_size)))
#Dropout to reduce overfitting
model.add(Dropout(0.5))
model.add(Conv1D(100, kernel_size=5, strides=1, activation='relu', padding='same'))
#Maxpooling to downsize matrix
model.add(MaxPooling1D(2))

model.add(Conv1D(100, kernel_size=5, strides=1, activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv1D(100, kernel_size=5, strides=1, activation='relu', padding='same'))

model.add(MaxPooling1D(2))

#Flatten 2D matrix to 1D vector
model.add(Flatten())

#Final fully-connnected layer
model.add(Dense(200))
'''
In 3-class classification, the following should be revised
'''
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Fit the model
model.fit(train_DL_x, train_DL_y,
          batch_size=20,
          shuffle=True,
          epochs=100,
          validation_split=0.1,
          #validation_data=(test_DL_x, test_DL_y),
          #min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement. 
          #patience: number of epochs with no improvement after which training will be stopped.
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00025, patience=2)])


'''
In 3-class classification, the following should be revised
'''
test_DL_pred_y = ['negative' if prediction[0] > prediction[1] else 'positive' for prediction in model.predict(test_DL_x)]


test_DL_y = processed_df['sentiment'][train_size:, ]

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(test_DL_y, test_DL_pred_y)
accuracy_score(test_DL_y, test_DL_pred_y)


'''
Q1: Predict with Airline dataset, which is a 3-class dataset
'''



'''
Q2(selective): If you have learned CNN before, try to adjust the model to see if you can get a better performance 
'''



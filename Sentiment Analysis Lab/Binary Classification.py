### Preprocess Text ###
from time import time
import numpy as np
import string
from basic_functions import *

import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
""" Tokenizes a text to its words, removes and replaces some of them """    

stoplist = stopwords.words('english')
my_stopwords = "st rd nd th am pm" # my extra stopwords
stoplist = stoplist + my_stopwords.split()
allowedWordTypes = ["J","R","V","N"] #  J is Adject, R is Adverb, V is Verb, N is Noun. These are used for POS Tagging
lemmatizer = WordNetLemmatizer() # set lemmatizer
stemmer = PorterStemmer() # set stemmer
processed_text = [] #used to store the preprocessed text and sentiment



def tokenize(text, textID, y):
    
    # tokens of one sentence each time
    onlyOneSentenceTokens = [] 
    
    # remove punctuation
    #e.g. text = "To be honest, I am OK.",  text.translate(translator) = "To be honest I am OK"
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator) 

    # it takes a text as an input and provides a list of every token in it
    #e.g. text = "To be honest, I am OK.",  nltk.word_tokenize(text) =  ['To', 'be', 'honest', ',', 'I', 'am', 'OK', '.']    
    tokens = nltk.word_tokenize(text) 
    
    # finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym
    #e.g. replaceNegations(["not","happy"]) = ["unhappy"]
    tokens = replaceNegations(tokens)
    
    # part of speech tagging  
    #e.g. token = ['To', 'be', 'honest', 'I', 'am', 'OK']  
    #     nltk.pos_tag(tokens) = [('To', 'TO'), ('be', 'VB'), ('honest', 'JJS'), ('I', 'PRP'), ('am', 'VBP'),('OK', 'JJ')]                
    tagged = nltk.pos_tag(tokens) 
    
    for w in tagged:

        if (w[1][0] in allowedWordTypes and w[0] not in stoplist):
            final_word = w[0].lower()
            
            #e.g. replaceElongated("laaaaaugh") = "laugh"
            final_word = replaceElongated(final_word)
            
            #e.g. lemmatizer.lemmatize("cats") = "cat"  lemmatizer.lemmatize("plays") = "play"
            final_word = lemmatizer.lemmatize(final_word)
            
            #e.g. stemmer.stem("cat's") = "cat"  stemmer.stem("joyful") = "joy"
            final_word = stemmer.stem(final_word)
            
                
            onlyOneSentenceTokens.append(final_word)           

         
    onlyOneSentence = " ".join(onlyOneSentenceTokens) # form again the sentence from the list of tokens
    processed_text.append(str(textID)+"\t"+ str(y)+"\t"+onlyOneSentence)
    
    """ Write the preprocessed text to file """
     


def pre_process(text,textID, y):
    text = removeUnicode(text) 
    text = removeURL(text) 
    text = removeAtUser(text) 
    text = removeHashtagInFrontOfWord(text) 

    
    text = replaceSlang(text) # replaces slang words and abbreviations with their equivalents
    text = replaceContraction(text) # replaces contractions to their equivalents
    text = removeNumbers(text) # remove integers from text

    text = replaceEmoticons(text)
    text = removeEmoticons(text) # removes emoticons from text
    text = replaceMultiExclamationMark(text) # replaces repetitions of exlamation marks with the tag "multiExclamation"
    text = replaceMultiQuestionMark(text) # replaces repetitions of question marks with the tag "multiQuestion"
    text = replaceMultiStopMark(text) # replaces repetitions of stop marks with the tag "multiStop"    
    tokenize(text, textID, y)  

'''
The dataset is from the weblink  
    https://www.kaggle.com/crowdflower/twitter-airline-sentiment/kernels 
'''

import pandas as pd
df = pd.read_csv("./twitter-airline-sentiment/Tweets.csv",header=0)

# shuffle rows of dataframe
df = df.sample(frac = 1, random_state = 100)

df = df[['airline_sentiment', 'text']]
df.rename(columns={'airline_sentiment':'sentiment'}, inplace=True)





'''
The dataset is from the weblink  
    https://www.kaggle.com/kazanova/sentiment140/version/2




import pandas as pd
import random as rnd

# randomly read s rows
n = 1600000
s = 160000
skip = sorted(rnd.sample(range(n), n-s))
df = pd.read_csv("./sentiment140/training.1600000.processed.noemoticon.csv",header=None,skiprows = skip)

df = df.iloc[:,[0,5]]
df.columns = ['sentiment','text']
df.loc[df['sentiment'] == 4 ,'sentiment'] = 'positive'
df.loc[df['sentiment'] == 0 ,'sentiment'] = 'negative'

'''


from tqdm import tqdm

# Get the list of processed_text
t0 = time()

for textID in tqdm(df.index):       
    y = df.loc[textID,'sentiment'] 
    text = df.loc[textID,'text']     
    pre_process(text, textID, y)
    
t1 = time()
print('\n Total processing time', round(t1-t0, 2),'s')



# Convert the processed_text from list to dataframe
import pandas as pd     
processed_text = [line.split('\t') for line in processed_text]
processed_text = pd.DataFrame(processed_text)
processed_text.rename(columns={0:'ID',1:'sentiment',2:'text'},inplace=True)
    


    
### Lexicon based methods ###

''' 
Some reference links about vaderSentiment:
    https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f
    https://github.com/cjhutto/vaderSentiment#about-the-scoring
'''

# vaderSentiment is an handy package, input the raw tweets is OK. No training set is needed.
# construct test set for vaderSentiment. The tweets in test_vad_x are raw tweets rather than processed tweets
test_vad_x = df['text'][int(0.8*len(df)):len(df)]
test_vad_y = df['sentiment'][int(0.8*len(df)):len(df)]
    
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

# output of analyser is polarity score, and we need to set rules to change the score to sentiment 'positive','negative','neutral'
# typical critical values are 0.05 and -0.05
def vader_to_sentiment(polarity_score):
    if polarity_score >= 0.05:
        sentiment = 'positive' 
    elif polarity_score <= -0.05:
        sentiment = 'negative' 
    else:
        sentiment = 'neutral' 
    return sentiment


# get predictions and evaluate the performance
test_vad_pred_y= [vader_to_sentiment(analyser.polarity_scores(i)['compound']) for i in test_vad_x]

from sklearn.metrics import confusion_matrix
confusion_matrix(test_vad_y, test_vad_pred_y)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(test_vad_y, test_vad_pred_y)



### Machine Learning Methods ### 

# Construct training set and test set
train_ML_x = processed_text['text'][:int(0.8*len(processed_text))]
train_ML_y = processed_text['sentiment'][:int(0.8*len(processed_text))] 
    
test_ML_x = processed_text['text'][int(0.8*len(processed_text)):]
test_ML_y = processed_text['sentiment'][int(0.8*len(processed_text)):] 


'''
The reference link to the machine learning methods:
    https://nlpforhackers.io/sentiment-analysis-intro/
'''


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# use count as value of features
unigram_bigram_clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer="word",
                                   ngram_range=(1, 2)# use unigram and bigram as features
                                   )),
    ('classifier', LinearSVC()) # LinearSVC means support vector machine classifier with linear kernel
])
unigram_bigram_clf.fit(train_ML_x, train_ML_y)
unigram_bigram_clf.score(test_ML_x, test_ML_y)
 


# use tfidf as value of features
unigram_bigram_clf = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer="word",
                                   ngram_range=(1, 2) # use unigram and bigram as features
                                  )),
    ('classifier', LinearSVC())# LinearSVC means support vector machine classifier with linear kernel
])
unigram_bigram_clf.fit(train_ML_x, train_ML_y)
unigram_bigram_clf.score(test_ML_x, test_ML_y)
 


### Deep Learning Methods ###

'''
The reference link about the deep learning methods:
    https://gist.github.com/giuseppebonaccorso/061fca8d0dfc6873619efd8f364bfe89
'''
import keras.backend as K
import multiprocessing
import tensorflow as tf

from gensim.models.word2vec import Word2Vec

# Create Word2Vec for tokens in the context of current processed text

vector_size = 300
window_size = 15
tokenized_text = [nltk.word_tokenize(u) for u in processed_text["text"]] 

word2vec = Word2Vec(sentences= tokenized_text,
                    size=vector_size, 
                    window=window_size, #The maximum distance between the current and predicted word
                    negative=20, # negative sampling ratio
                    iter=500, # number of iteration over the corpus
                    seed=1000,
                    workers=multiprocessing.cpu_count())

X_vecs = word2vec.wv
del word2vec


train_size = int(0.8*len(processed_text))
test_size = len(processed_text) - train_size
max_tweet_length = 20 # maximum number of tokens used in each sentence

# Construct train and test sets
# Generate random indexes
#indexes = set(np.random.choice(len(processed_text), len(processed_text), replace=False))

train_DL_x = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
train_DL_y = np.zeros((train_size, 2), dtype=np.int32)
test_DL_x = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
test_DL_y = np.zeros((test_size, 2), dtype=np.int32)



for i in range(len(processed_text)):
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
        train_DL_y[i, :] = [1.0, 0.0] if processed_text['sentiment'][i] == "negative" else [0.0, 1.0]
    else:
        test_DL_y[i - train_size, :] = [1.0, 0.0] if processed_text['sentiment'][i] == "negative" else [0.0, 1.0]
  


import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam


# Set random seed (for reproducibility)
np.random.seed(1000)

# Select whether using Keras with or without GPU support
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
#use_gpu = True
#
#config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(), 
#                        inter_op_parallelism_threads=multiprocessing.cpu_count(), 
#                        allow_soft_placement=True, 
#                        device_count = {'CPU' : 1, 
#                                        'GPU' : 1 if use_gpu else 0})
#
#session = tf.Session(config=config)
#K.set_session(session)

# Keras convolutional model
batch_size = 10
nb_epochs = 100

model = Sequential()

model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(max_tweet_length, vector_size)))
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='relu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='relu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='relu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='relu', padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Fit the model
model.fit(train_DL_x, train_DL_y,
          batch_size=batch_size,
          shuffle=True,
          epochs=nb_epochs,
          validation_data=(test_DL_x, test_DL_y),
          #min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement. 
          #patience: number of epochs with no improvement after which training will be stopped.
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00025, patience=2)])





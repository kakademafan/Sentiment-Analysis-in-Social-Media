#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:03:54 2019

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


'''
In 3-class classification, the following should be revised
'''
# output of analyser is polarity score between -1 (most extreme negative) and +1 (most extreme positive)
def vader_to_sentiment(polarity_score):
    if polarity_score >= 0.0:
        sentiment = 'positive' 
    else:
        sentiment = 'negative' 

    return sentiment



# get predictions and evaluate the performance
test_vad_pred_y= [vader_to_sentiment(analyser.polarity_scores(i)['compound']) for i in test_vad_x]


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(test_vad_y, test_vad_pred_y)
accuracy_score(test_vad_y, test_vad_pred_y)


'''
Q1: Predict with Airline dataset, which is a 3-class dataset 
'''
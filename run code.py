# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import linear_model, datasets
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
import pandas as pd,re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.cluster import KMeansClusterer, euclidean_distance
import matplotlib.pyplot as plt
		


df = pd.read_csv("ZARA.csv",sep=";") 
# df_negative = df[df['vader_sentiment'] == 'Negative'][['cleaned_tweet','vader_sentiment']].head(1000)
df = df[df['textblob_sentiment'] == df['vader_sentiment']]
df_negative = df[df['vader_sentiment'] == 'Negative'][['cleaned_tweet','vader_sentiment']].head(1000)
df_positive = df[df['textblob_sentiment'] == 'Positive'][['cleaned_tweet','vader_sentiment']].head(1000)
df_neutral = df[df['textblob_sentiment'] == 'Neutral'][['cleaned_tweet','vader_sentiment']].head(1000)

df = pd.concat([df_negative,df_positive,df_neutral])
df = df.rename(columns={'vader_sentiment': 'Tag','cleaned_tweet':'Text'})
df = df.sample(frac=1)
df=df.dropna()
train_split =0.7


result=[]


for i in range(1,100,3):
	text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=i)) ])
	text_clf = text_clf.fit(df['Text'][:int(len(df)*train_split)], df['Tag'][:int(len(df)*train_split)])
	predicted = text_clf.predict(df['Text'][int(len(df)*train_split)+1:])
	print i
	result.append({"parameter":i,"accuracy":metrics.accuracy_score(df['Tag'][int(len(df)*train_split)+1:], predicted)*100 })
	
temp = pd.DataFrame(result)
plt.plot(temp['parameter'],temp['accuracy'])
plt.ylabel('Accuracy (%)') 
plt.xlabel('Laplace smoothing parameter') 
plt.title('Accuracy(%) Vs Laplace smoothing parameter') 
plt.show()

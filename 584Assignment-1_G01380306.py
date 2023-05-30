#!/usr/bin/env python
# coding: utf-8

# In[325]:


import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=1000)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import re
import nltk
from scipy.spatial import distance_matrix

from nltk.corpus import stopwords 
from numpy.linalg import norm

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cityblock
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


# In[326]:


#Training Data path -> "/Users/sanjanagovindu/Downloads/DM1/train_file.csv"
#Test Data path -> "/Users/sanjanagovindu/Downloads/DM1/test_file.csv"

train=pd.read_fwf("/Users/sanjanagovindu/Downloads/DM1/train_file.csv",colspecs=[(0,2),(3,None)],names=["sentiment", "review"],delimiter="#EOF")
test=pd.read_fwf("/Users/sanjanagovindu/Downloads/DM1/test_file.csv",names=["review"],delimiter="\n")

train_sentiments=train["sentiment"].tolist()
train_reviews=train["review"].tolist()
test_reviews=test["review"].tolist()


# In[327]:


def preprocessing(df):
    
    #Convert data to lower case
    df['cleanReview'] = df['review'].str.lower() 
    #Remove punctuations
    df['cleanReview'] = df['cleanReview'].str.translate(str.maketrans('', '', string.punctuation))
    #Remove numbers
    df['cleanReview'] = df['cleanReview'].str.replace('\d+', '')
    #Remove HTML Tags
    df['cleanReview'] = df['cleanReview'].str.replace(r'<[^<>]*>', '', regex=True)
    #Remove URLs
    df['cleanReview'] = df['cleanReview'].str.replace('https?://S+|www.S+', '', regex=True)
    #Remove white spaces
    df['cleanReview'] = df['cleanReview'].str.strip()
    #Spelling Correction
    
    #Remove stop words
    stop_words = set(stopwords.words('english'))
    df['cleanReview'] = df['cleanReview'].apply(lambda x: ' '.join([w for w in x.split() if w not in (stop_words)]))
    
    #Split words
    df['cleanReview'] = df['cleanReview'].str.split()
    
    #Tokenization
#     df['cleanReview'] = df.apply(lambda row: nltk.word_tokenize(row['cleanReview']), axis=1)
    
    #Stemming
    porter_stemmer = PorterStemmer()
    df['cleanReview'] = df['cleanReview'].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    
    #Lemmetization
#     lemmatizer = nltk.stem.WordNetLemmatizer()
#     df['cleanReview'] = df['cleanReview'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    
    cleanedData=df['cleanReview'].tolist()
    for i in range(0,len(cleanedData)):
        cleanedData[i]=' '.join(cleanedData[i])
    return cleanedData


# In[328]:


def prediction(nearest_neighbors, sentiments):    
    pos= 0
    neg= 0
    for neighbor in nearest_neighbors:
        if int(sentiments[neighbor]) == 1:
            pos+= 1
        else:
            neg+= 1
    #Calculate max of postives and negatives to predict the test sentiments
    if max(pos,neg)==pos:
        return 1
    else:
        return -1


# In[329]:


def calculateDistance(test_vector,train_vector):
    #Cosine Similarity
    distance=cosine_similarity(test_vector,train_vector)
    
     #Euclidean distance
#     distance=euclidean_distances(test_vector,train_vector)

     #Manhattan distance
#     distance = cityblock(test_vector,train_vector)
    return distance


# In[330]:


def KNN(train_vector,test_vector,train_sentiments,k):
    
    #Calculate distance - Cosine similarity
    distance=calculateDistance(test_vector,train_vector)

    test_sentiments = []
    for d in distance:
        #returns kNN indices
        knn = np.argsort(d)[::-1][:k]
        prediction = predict(knn, train_sentiments)
        test_sentiments.append(1) if prediction == 1 else test_sentiments.append(-1)

    return test_sentiments


# In[331]:


def joinAll(arr):
    for i in range(0,len(arr)):
        arr[i]=' '.join(arr[i])
    return arr


# In[332]:


#Preprocessing both training data and testing data to create simarity vector
train_reviews = preprocessing(train)
test_reviews = preprocessing(test)

#TFIDF Vectorization - converts training and test data list to vectors
vectorizer = TfidfVectorizer()
train_vector = vectorizer.fit_transform(train_reviews)
test_vector = vectorizer.transform(test_reviews)

#SVD (Singular value decomposition) - Factorization of a matrix.
# train_vector=svd.fit_transform(train_vector)
# test_vector=svd.transform(test_vector)


# In[333]:


#K-Fold Cross Validation
kfold = KFold(n_splits=11, random_state=None)
x=train['cleanReview']
y=train['sentiment']
accuracies = []
 
for train_index , test_index in kfold.split(x):
    x_train , x_test = x.iloc[train_index],x.iloc[test_index]
    y_train , y_test = y[train_index] , y[test_index]
    x_train=x_train.tolist()
    y_train=y_train.tolist()
    x_test=x_test.tolist()
    
    train_reviews1, test_reviews1=joinAll(x_train),joinAll(x_test)
    
    #TFIDF Vectorization
    vectorizer = TfidfVectorizer()
    train_vector1 = vectorizer.fit_transform(train_reviews1)
    test_vector1 = vectorizer.transform(test_reviews1)
    
    a = accuracy_score(pd.Series(KNN(train_vector1,test_vector1,y_train,255)) , y_test)
    accuracies.append(a)

print('Accuracy of each fold - {}'.format(accuracies))
avg_accuracy = sum(accuracies)/k
print('Avg accuracy : {}'.format(avg_accuracy))


# In[334]:


test_sentiments=KNN(train_vector,test_vector,train_sentiments,255)
#Writing the output which is test sentiments to an output text file
fout = open('/Users/sanjanagovindu/Downloads/DM1/output.txt', 'w')
fout.writelines( "%s\n" % i for i in test_sentiments)
fout.close()


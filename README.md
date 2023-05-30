# Movie-Review-Sentiment-Analysis-using-KNN

**Objectives of the assignment:**
- Implementation of the KNN algorithm
- Handle and preprocess the movie reviews dataset which has been provided
- Choose the best KNN model with features and distance functions
- Implement cross validation algorithm on KNN to calculate the accuracy score

**Problem statement:**
The assignment is about Movie review sentiment analysis. In this assignment, provided the training and test reviews along with the train sentiments. We had to implement the KNN classifier algorithm to predict the test sentiments for 18000 rows or reviews of data. Positive sentiments are represented +1 and negative sentiment are by -1. A test file will be provided in the assignment which can be used to compare with the predictions.

**APPROACH FOLLOWED:**
1. Loaded the training dataset and the test data set into data frames using the read fwf in python. (Provided local file paths)
2. Data has been preprocessed using various methods. The following process has been done in the preprocessing method:
    - Data has been converted to lower case
    - Punctuations has been removed from the data
    - Numbers have been removed
    - HTML tags like r'<[^<>]*>' have been removed
    - URLs like 'https?://S+|www.S+' have been removed
    - White spaces has been removed from the data
3. Moreover, stops words in English has been removed to increase the accuracy rate in the data. The special characters like punctuations and stop words removal play a huge role in accuracy calculation in this implementation.
4. After that, the data has been split into list of words.
5. Then, the data has to be stemmed/lemmatized. But stemming process has been followed because lemmatization process gave less accuracy compared to stemming.
6. For vectorization as with TFIDF it is 0.85 and bag of words is 0.78.
7. Below are my parameters used in the experiment:
    - Neighbors count: value of k is 255
    - TFIDF Vectorization - converts training and test data list to vectors
    - Cosine similarity – used to calculate distance between 2 vectors
    - K fold cross validation with k value as 11 gave accuracy as 0.85.
8. The reasons for choosing the above parameters:
    - K value – number of neighbors: The value of k is 255 has been choose as it increased the accuracy score for the implementation. Many k values       has been considered, and out of them the best k value which gave highest accuracy has been choose as the final one.
    - Cosine similarity has been considered here because Cosine similarity vs Euclidean vs Manhattan distance, all have been compared and cosine         similarity gave the highest accuracy which is 0.85.
    - SVD is used mostly for sparse data and reduces the dimensions by decreasing the input variables of dataset. Without SVD the accuracy is 0.85        and with SVD it is 0.798 and the runtime will be reduced if SVD is used.
  
  <img width="368" alt="image" src="https://github.com/sanjana-govindu/Movie-Review-Sentiment-Analysis-using-KNN/assets/54507596/70a910e1-d6da-4fdd-a75b-7210f3d682bc">

  

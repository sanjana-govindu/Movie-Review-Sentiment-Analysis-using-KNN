# Movie-Review-Sentiment-Analysis-using-KNN

**Objectives of the assignment:**
- Implementation of the KNN algorithm
- Handle and preprocess the movie reviews dataset which has been provided
- Choose the best KNN model with features and distance functions
- Implement cross validation algorithm on KNN to calculate the accuracy score

**Problem statement:**
The assignment is about Movie review sentiment analysis. In this assignment, provided the training and test reviews along with the train sentiments. We had to implement the KNN classifier algorithm to predict the test sentiments for 18000 rows or reviews of data. Positive sentiments are represented +1 and negative sentiment are by -1. A test file will be provided in the assignment which can be used to compare with the predictions.

**Detailed Description:**
A practical application in e-commerce applications is to infer sentiment (or polarity) from free-form review text submitted for a range of products. This is to implement a k-Nearest Neighbor Classifier to predict the sentiment for 18,000 reviews for various products provided in the test file (test_file.csv). 
- Positive sentiment is represented by a review rating of +1 and negative sentiment is represented by a review rating of -1. 
- In the test file, we are only provided the reviews but no ground truth rating which will be used for comparing your predictions. 
- Training data consists of 18000 reviews and exists in the file train_file.csv. 
- Each row begins with the sentiment score followed by the text of the review.  
- format.csv shows an example file containing 18000 rows alternating with +1 and -1.

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
  
  
  
**During the pre-processing of the original dataset, the following steps were taken into consideration:**

Positive and negative classification: Reviews with a score of 3 or 4 were classified as positive, while those with a score of 0 or 1 were classified as negative. Reviews with a score of 2 (considered neutral) were ignored for the time being. 

Removal of duplicate entries: Duplicate reviews in the dataset were eliminated. Four featurizations for training: The final model was trained using four featurizations:

a. Bag of Words (BoW): This approach converts words in reviews into vectors. Each unique word in the dataset becomes a dimension, and the count of each word in a review is placed in the corresponding dimension. This results in a sparse matrix representation of the words in the reviews.

b. TF-IDF (Term Frequency-Inverse Document Frequency): TF calculates the frequency of a word in a review divided by the total number of words in that review. IDF calculates the rarity of a word across the entire corpus. The product of TF and IDF values is computed for each word in each review, resulting in a d-dimensional vector representation.

c. Average Word-to-Vec (Avg W2V): Each word in a review is converted into a vector representation. The average vector of all the words in a review is computed.

d. TF-IDF Weighted Word-to-Vec: Similar to Avg W2V, each word in a review is converted into a vector representation. The TF-IDF weighted average vector of all the words in a review is computed.

**Preprocessing of reviews:** The reviews underwent several preprocessing steps:

- Removal of HTML tags.
- Removal of punctuations and limited special characters.
- Checking if the word consists of English letters and is not alphanumeric.
- Checking if the word length is greater than 2, as adjectives are not typically 2 letters long.
- Conversion of words to lowercase.
- Removal of stopwords.
- Snowball Stemming, which is observed to be more effective than Porter Stemming
- Splitting the data: After featurization, the data was split to assess the model's performance on unseen data
- Training different models: Various models such as KNN, Naive Bayes, Logistic Regression, SVMs, Decision Tree, and Random Forest were trained and evaluated.

In summary, the preprocessing steps involved removing HTML tags, punctuation, and special characters, checking word characteristics, converting to lowercase, removing stopwords, and applying stemming. The featurizations included Bag of Words, TF-IDF, Average Word-to-Vec, and TF-IDF Weighted Word-to-Vec. The data was split for evaluation, and different models were trained and tested.


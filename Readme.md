# Analysis of Similarity Measurement Techniques on Text Documents

## Introduction
This project aims to investigate and compare different methods for measuring similarities between text documents. The main objective of the project is to explore various similarity metrics and analyse their effectiveness in capturing the similarity relationships between documents.

The different similarity measurement techniques we have worked on are:

* Cosine
* Jaccard
* Euclidean
* SimRank
* Jenson Shennon 
* Dice
* Manhatten

## Dataset Description

The dataset I used to check the similarities between different documents is the well-known 20 Newsgroups dataset.

This dataset contains 20 text files. To measure similarity, I started with the cosine similarity method. I first compared how close text file 1 is to text file 2, then compared text file 1 to text file 3, and so on for all the files. I repeated this process using the Jaccard index and other similarity methods.

You can find the dataset [here](https://www.kaggle.com/datasets/crawford/20-newsgroups).

### Data preprocessing steps:
* **Data cleaning** : Like in this code we did not really removed any special characters, punctuation marks or numbers. However during file process we used *encoder = ‘utf-8’* parameter to handle potential encoding issues and the *errors = ‘ignore’* parameter to ignore any encoding issues.

* **Stopword Removal** : In this code we didn’t included any code to remove stop words but when we used “CountVectorizer” stopwords removal applied implicitly based on the predefined stopwords list.

* **Tokenization** : The code tokenizes the documents implicitly during the vectorization process using the CountVectorizer from the scikit-learn library. The vectorizer splits the text into individual words or tokens using its default tokenizer.

* **Vectorization** : The code utilizes the CountVectorizer from scikit-learn to convert the preprocessed text into numerical representations. The fit_transform() method of the CountVectorizer is used to transform the text into a document-term matrix, which represents the frequency of words in each document.







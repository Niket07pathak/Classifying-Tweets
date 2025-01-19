
# **Tweet Classifier**

This project focuses on classifying tweets based on the political affiliation of their authors, using supervised machine learning techniques. The tweets, extracted via the Twitter API, pertain to the 2016 US Presidential election and are sourced from prominent accounts like `realDonaldTrump`, `HillaryClinton`, `GOP`, `TheDemocrats`, and others. The goal is to classify tweets as Republican or Democratic based on their content.

---

## **Project Overview**

The project is structured into three main phases:
1. **Text Processing**: Cleaning and normalizing raw tweets using techniques such as tokenization, lemmatization, and removal of stop words, punctuations, and URLs.
2. **Feature Construction**: Building bag-of-words TF-IDF feature vectors from processed tweets and deriving labels based on the authors' affiliations.
3. **Classification**: Training and evaluating Support Vector Machine (SVM) classifiers with different kernels to achieve optimal tweet classification.

---

## **Features**

### **Data Preprocessing**
- Tokenization using `nltk.word_tokenize`.
- Lemmatization based on part-of-speech tagging.
- Removal of URLs, special characters, and unnecessary punctuations.
- Custom stop word handling tailored to the dataset.

### **Feature Engineering**
- TF-IDF vectorization of tweet content with `sklearn.feature_extraction.text.TfidfVectorizer`.
- Filtering out rare words and ignoring common stop words.

### **Classification**
- Implementation of a baseline Majority Label Classifier.
- SVM classifiers with kernels (`linear`, `poly`, `rbf`, `sigmoid`) trained using `sklearn.svm.SVC`.
- 4-fold cross-validation to evaluate kernel performance.
- Automated pipeline to classify unseen tweets from test data.

---

### **Requirements**
- Python 3.x
- Required Python libraries: `nltk`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.


## **Results**
- **Baseline Accuracy**: 50.01% (Majority Label Classifier)
- **SVM Accuracy**: Achieved up to 95.4% with the linear kernel, outperforming the baseline significantly.

---



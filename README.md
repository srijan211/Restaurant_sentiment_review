# Restaurant Review Sentiment Analysis

This project implements a sentiment analysis model that classifies restaurant reviews as **positive** or **negative**. The model uses the Naive Bayes classifier, a simple yet effective machine learning algorithm. The dataset consists of 1,000 restaurant reviews, and the code preprocesses, transforms, and classifies these reviews based on their content.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Libraries Used](#libraries-used)
4. [Code Explanation](#code-explanation)
    1. [Data Preprocessing](#data-preprocessing)
    2. [Model Training](#model-training)
    3. [Model Evaluation](#model-evaluation)
    4. [Hyperparameter Tuning](#hyperparameter-tuning)
    5. [Prediction](#prediction)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Project Overview

This project focuses on classifying restaurant reviews as positive or negative based on the content of the review. The model uses Natural Language Processing (NLP) techniques to preprocess the reviews and then applies a **Multinomial Naive Bayes** classifier for sentiment analysis. The dataset contains 1,000 reviews and their corresponding sentiment labels (1 for positive, 0 for negative).

## Dataset

The dataset used in this project is a tab-separated values file (`Restaurant_Reviews.tsv`) containing two columns:

1. **Review**: The restaurant review text.
2. **Liked**: A binary label indicating the sentiment of the review (1 for positive, 0 for negative).

The dataset contains a total of 1,000 rows.

## Libraries Used

This project uses the following libraries:

- **pandas**: For data manipulation and loading the dataset.
- **nltk**: For natural language processing tasks (e.g., tokenization, stopword removal, stemming).
- **sklearn**: For machine learning tasks, including vectorization, model training, and evaluation.
- **numpy**: For numerical operations.
- **re**: For regular expressions used in text preprocessing.

You can install the required libraries using the following command:

```bash
pip install pandas nltk scikit-learn numpy
```

## Code Explanation

### Data Preprocessing

Before training the model, the reviews are preprocessed to convert the text into a format suitable for machine learning. The following steps are applied:

1. **Removing special characters**: Any non-alphabetic characters are removed from the review text.
2. **Lowercasing**: All text is converted to lowercase to maintain uniformity.
3. **Tokenization**: The review text is split into individual words.
4. **Stopword Removal**: Common words such as "the", "and", "is", etc., that do not contribute to the sentiment, are removed.
5. **Stemming**: Words are reduced to their base or root form (e.g., "running" becomes "run").

The cleaned text is then stored in a **corpus** for later vectorization.

### Model Training

The **CountVectorizer** from scikit-learn is used to transform the text data into a numeric form that can be fed into the machine learning model. The `max_features=1500` parameter limits the features to the top 1500 most frequent words in the reviews.

A **Multinomial Naive Bayes** classifier is then trained on the transformed training data. This model is effective for text classification tasks like sentiment analysis, where the goal is to predict a category based on word frequencies.

### Model Evaluation

The trained model is evaluated on the test set using the following metrics:

- **Accuracy**: The overall percentage of correctly predicted reviews.
- **Precision**: The percentage of positive predictions that are correct.
- **Recall**: The percentage of actual positive reviews that are correctly identified.
  
Additionally, a **confusion matrix** is generated to show the true positives, false positives, true negatives, and false negatives.

### Hyperparameter Tuning

The Naive Bayes classifier uses an **alpha** parameter, which helps with smoothing probabilities. The optimal value of alpha is determined by testing values in the range of 0.1 to 1.0. The best accuracy is achieved with **alpha=0.2**.

### Prediction

After training and evaluating the model, you can use it to predict the sentiment of new restaurant reviews. The `predict_sentiment` function processes a sample review, transforms it, and uses the trained classifier to predict whether the review is positive or negative.

Example usage:

```python
sample_review = 'The food is really good here.'
if predict_sentiment(sample_review):
    print('This is a POSITIVE review.')
else:
    print('This is a NEGATIVE review!')
```

## Results

### Accuracy, Precision, and Recall

The model achieves the following scores on the test dataset:

- **Accuracy**: 76.5%
- **Precision**: 0.76
- **Recall**: 0.79

### Confusion Matrix

The confusion matrix for the test set is as follows:

```
[[72, 25]
 [22, 81]]
```

- **True Positives (TP)**: 81
- **False Positives (FP)**: 25
- **True Negatives (TN)**: 72
- **False Negatives (FN)**: 22

### Best Alpha Value

After tuning the **alpha** parameter, the best accuracy was found with **alpha=0.2**, which gave an accuracy of 78.5%.

## Conclusion

The Naive Bayes classifier performs well in classifying restaurant reviews as either positive or negative. The preprocessing steps, including stopword removal and stemming, helped improve the quality of the text data. Hyperparameter tuning was also effective in improving model performance, resulting in a best accuracy of 78.5%.

This sentiment analysis model can be further enhanced by using more advanced techniques like TF-IDF vectorization or deep learning models, and it can be deployed to classify reviews in real-time.

---

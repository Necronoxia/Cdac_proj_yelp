# Sentiment analysis and rating prediction for customer reviews

## Synopsis:
This project seeks to solve the problem faced by businesses that have review/comment systems but due to the high
volume of data, they cannot manage to analyze all the comments and generate insights that can be useful for the
business. We do this by using machine learning-based natural language processing techniques to predict the rating
and also generate the type of sentiment that is being conveyed by the text. Along with this, we provide the topic that
is being discussed using topic modeling.

## Notebook sequence:
Sequence of notebooks to execute for final transformed data output.
### Rating prediction:
1. ../pipeline/Preprocessing-pipeline-Grams.ipynb
2. ../models/RatingClassifierNgram.ipynb

### Sentiment analysis:
1. ../models/positive_negative_final.ipynb

### Topic Modelling:
1. ../Topic_Modelling.py (Demo-version)
2. ../TOPIC MODELLING FINAL.ipynb (Full version)

## Architecture:
![project_architechture_final ](https://user-images.githubusercontent.com/41346159/162556455-45e60f22-6d07-4fce-9c30-d710542689af.png)

## Observations:
|          Metrics            | One Vs Rest Unigram | One Vs Rest Bigram | One Vs Rest Trigram | Logistic regression |
|:---------------------------:|:-------------------:|:------------------:|:-------------------:|:-------------------:|
|     Training Accuracy       | 0.5984332967644009  | 0.5485745630513518 | 0.36941284320944984 | 0.8641047351868575  |
|     Training F1-score       | 0.590775703312456   | 0.5418588057916316 | 0.3644811915058533  | 0.8380442784806515  |
|  Training weightedPrecision | 0.5896686737826645  | 0.5420593428000069 | 0.36945976463769    | 0.8699591633878737  |
|   Training weightedRecall   | 0.5984332967644009  | 0.5485745630513518 | 0.3694128432094498  | 0.8641047351868575  |
|      Testing Accuracy       | 0.5656358333916001  | 0.4966641330016262 | 0.30505364678604213 | 0.8629278212689175  |

# Sentiment analysis and rating prediction for customer reviews

## Synopsis:
This project seeks to solve the problem faced by businesses that have review/comment systems but due to the high
volume of data, they cannot manage to analyze all the comments and generate insights that can be useful for the
business. We do this by using machine learning-based natural language processing techniques to predict the rating
and also generate the type of sentiment that is being conveyed by the text. Along with this, we provide the topic that
is being discussed using topic modeling.

## Architecture:
![project_architechture_final  drawio](https://user-images.githubusercontent.com/41346159/162418183-584daa19-0150-4d34-a2f4-758d13827042.png)

## Observations:
|          Metrics            | One Vs Rest Unigram | One Vs Rest Bigram | One Vs Rest Trigram | Logistic regression |
|:---------------------------:|:-------------------:|:------------------:|:-------------------:|:-------------------:|
|     Training Accuracy       | 0.5984332967644009  | 0.5485745630513518 | 0.36941284320944984 |                     |
|     Training F1-score       | 		                | 0.5418588057916316 | 0.3644811915058533  |                     |
|  Training weightedPrecision | 		                | 0.5420593428000069 | 0.36945976463769    |                     |
|   Training weightedRecall   | 		                | 0.5485745630513518 | 0.3694128432094498  |                     |
|      Testing Accuracy       | 0.5656358333916001  | 0.4966641330016262 | 0.30505364678604213 |                     |

#######################################################################################################################
#############################  TOPIC MODELLING #######################################################################
#######################################################################################################################
import pyspark
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, Tokenizer, HashingTF, IDF
from pyspark.sql.functions import array
import nltk
from nltk.stem import PorterStemmer
from pyspark import keyword_only
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml import Transformer, classification
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from nltk.stem import PorterStemmer
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import * 
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
#import pandas as pd
import string 
import re
##############################################################
review_df= spark.read.parquet("s3://mukeshproj/Topic_modelling_pipeline_backup/topic_modelling_parquet_pipeline/shrink_data_piprline_backup")


################# Filtering Data #############################
df=review_df.filter(review_df.label=='1')
df_badreviews=df.limit(5000).toPandas()
stem=df_badreviews.stemmed.tolist()
##############################################################

import pandas as pd
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

##############################################################
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.decomposition import NMF

################################ Tokenize #####################
def tokenize(texts):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    texts_tokens = []
    for i, val in enumerate(texts):
        text_tokens = tokenizer.tokenize(val.lower())
        for i in range(len(text_tokens) - 1, -1, -1):
            if len(text_tokens[i]) < 4:
                del(text_tokens[i])
        texts_tokens.append(text_tokens)
    return texts_tokens
################################################################
#texts_tokens = tokenize(texts)
########################### Remove Stopwords####################
def removeSW(texts_tokens):
    stopWords = set(stopwords.words('english'))
    texts_filtered = []
    for i, val in enumerate(texts_tokens):
        text_filtered = []
        for w in val:
            if w not in stopWords:
                text_filtered.append(w)
        texts_filtered.append(text_filtered)
    return texts_filtered
###########################################################
#texts_filtered = removeSW(texts_tokens)
##########################Lemmitization####################
def lemma(texts_filtered):
    wordnet_lemmatizer = WordNetLemmatizer()
    texts_lem = []
    for i, val in enumerate(texts_filtered):
        text_lem = []
        for word in val:
            text_lem.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        texts_lem.append(text_lem)
    return texts_lem
###############################################################
#texts_lem = lemma(texts_filtered)
texts_string = []
for text in stem:
    string = ' '
    string = string.join(text)
    texts_string.append(string)
#######################Libraries installation#################
nltk.download('stopwords')
nltk.download('wordnet')
###########################################################

import pandas as pd
import json
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF


#############################################################

def plot_top_words(model, feature_names, n_top_words, title):
    #Modified from SKlearn
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 15})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=15)
        ax.tick_params(bottom=False)
        ax.set(xticklabels=[])

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    plt.savefig('/home/hadoop/Topics_in_LDA.png')

##############################################################
vectorizer = CountVectorizer(max_df=0.90, min_df=5)
##############################################################
X = vectorizer.fit_transform(texts_string)
feature_names =  vectorizer.get_feature_names()

X.toarray().shape
###############################################################
n_topics = 10

lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=12
)

lda.fit_transform(X)
##############################################################
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, feature_names, no_top_words)
########################Topics in LDA###########################

#plot_top_words(lda, feature_names, no_top_words,'Topics in LDa')

########################tfidf_vectorizer#########################

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, 
    min_df=5,  
    stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(texts_string)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

#################################################################
from sklearn.decomposition import NMF
#################################################################

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, 
    min_df=5,  
    stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(texts_string)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

##################################################################
from sklearn.decomposition import NMF
####################################################################

no_topics = 10

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')
nmf.fit_transform(tfidf)

#####################################################################

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)

######################################################################

plot_top_words(nmf, tfidf_feature_names, no_top_words,'Topics in NMF')

######################################################################

########################LDA visualization Libraries###########
from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
#######################CountVectorizer#################################
cv = CountVectorizer(inputCol="texts_string", outputCol="raw_features", vocabSize=5000, minDF=10.0)
dictionary = gensim.corpora.Dictionary(stem)
len(dictionary.cfs)
#######################bow_corpus######################################
bow_corpus = [dictionary.doc2bow(doc) for doc in stem]
tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
############################lda_model_coherence Graph###################
lda_model_coherence = []
for i in range (2,15):
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=i, id2word=dictionary, passes=5, workers=4)
    cm = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    lda_model_coherence.append(coherence)
import matplotlib.pyplot as plt
plt.plot(range(2, 15),lda_model_coherence)
plt.xlabel('Number of topics')
plt.ylabel('Coherence score')
plt.title('How many topics ? (Closer to 0 = better)')
plt.show()
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=5, workers=4)
vis = gensimvis.prepare(topic_model=lda_model, corpus=bow_corpus, dictionary=dictionary)
#pyLDAvis.enable_notebook()
pyLDAvis.display(vis)

plt.savefig('/home/hadoop/data.png')
pyLDAvis.save_html(vis, '/home/hadoop/lda.html')
############################################################################
tfidf = gensim.models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=5, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

vis = gensimvis.prepare(topic_model=lda_model_tfidf, corpus=corpus_tfidf, dictionary=dictionary)
#pyLDAvis.enable_notebook()
pyLDAvis.display(vis)
pyLDAvis.save_html(vis, '/home/hadoop/lda1.html')
############################################################################
#                INPUT SECTION
##############################################################################

def preprocess(raw_text):
    x = tokenize(raw_text)
    x = removeSW(x)
    x = lemma(x)
    return x

unseen_document = ['I had to wait 2 hours. It was so long. I will never come back here.', 'Food was horrible. My pizza was burn and was late.', 'It tasted bad. It is not good quality. Who is cooking here ?']
print(unseen_document)
preprocessed_doc = preprocess(unseen_document)
bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_doc]

print('There is', len(preprocessed_doc), 'reviews in the unseen document. We are going to calculate the score for the best topic of each of them.\n---------------\n')

for i in range(0, len(preprocessed_doc)):
    for index, score in sorted(lda_model[bow_corpus[i]], key=lambda tup: -1*tup[1]):
        print("# Review", i, ": best score: {}\t For topic: {}".format(score, lda_model.print_topic(index, 5)), '\n')
        break

###########################################################################

























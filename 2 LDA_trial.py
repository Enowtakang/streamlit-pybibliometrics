import gensim.utils
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora
import nltk
# nltk.download('wordnet')
import numpy as np
np.random.seed(2022)


"""
Load Data
"""
path = "C:/Users/HP/PycharmProjects/MachineLearningEnow/Streamlit_Pybibliometrics/eddy.csv"
data = pd.read_csv(path)
df = data["title"]
# print(df.head())

"""
Pre-processing
"""
stemmer = SnowballStemmer("english")


def lemmatize_stemming(text):
    return stemmer.stem(
        WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


"""
Map all documents through the 
'preprocess' function
"""
processed_docs = df.map(preprocess)
# print(processed_docs[:10])


"""
Transform data into a bag_of_words model.
    We would get a dictionary containing the
    number of times each word appears in the 
    training dataset.
    
    Notice that the ID, the WORD and ITS FREQUENCY 
    have been printed. 
    Also, the words are in alphabetical order.
"""
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0

for k, v in dictionary.iteritems():
    # feel free to uncomment the
    # commented code below:

    # print(k, v, dictionary.dfs[k])

    count += 1
    if count > 10:
        break

"""
Filter the tokens:

    I would NOT do this part 
    since my dataset is very small.
    
    However, the code below serves to remove
    tokens that occur in less that 15 documents
    (titles) or more than half of the documents
    (titles)  - here, this is if there are MORE
    THAN 1 MILLION TITLES!!!
"""
# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


"""
Create a bag_of_words (bow) corpus
"""
bow_corpus = [
    dictionary.doc2bow(doc) for doc in processed_docs]

# note this area
n = len(df)-1
bow_doc_n = bow_corpus[n]


def show_contents_bow_doc_n():
    print(bow_doc_n)
    for i in range(len(bow_doc_n)):
        print(
            "Word {} (\"{}\") appears {} time.".format(
                bow_doc_n[i][0],
                dictionary[bow_doc_n[i][0]],
                bow_doc_n[i][1]))


# show_contents_bow_doc_n()


"""
Lets create our 
Latent Dirichlet Association model.
NOTE: You are using windows: Don't use LdaMulticore
"""
lda_model = gensim.models.LdaModel(
    bow_corpus,
    # here, ask the user how many topics
    # he/she wants the corpus to be divided into.
    # let the user also specify the number of passes:
    # more passes means better results
    num_topics=5,
    id2word=dictionary, passes=4)


def show_topics():
    for idx, topic in lda_model.print_topics(-1):
        print(
            "Topic {} \nWords: {}".format(
                idx, topic))


show_topics()




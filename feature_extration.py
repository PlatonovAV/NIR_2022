from pathlib import Path

import mglearn as mglearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words_vizualizer():
    # загрузка данных из файла
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step2', 'train.csv')
    df_final_path = Path(work_path, 'dataset\\clear\\step3', 'train.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]

    # Представление через WordCloud
    corpus = get_corpus(df_a['ABSTRACT'].values)
    procWordCloud = get_wordCloud(corpus)
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(procWordCloud)
    plt.axis('off')
    plt.subplot(1, 2, 1)

    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=2000, min_df=5, max_df=0.7,
                                 stop_words=stopwords.words('english'))
    X_physics = vectorizer.fit_transform(df_a["ABSTRACT"])
    count_vect_df = pd.DataFrame(X_physics.todense(), columns=vectorizer.get_feature_names_out(), index=df_a.index)
    count_vect_df = count_vect_df.T
    count_vect_df["TOTAL"] = count_vect_df.sum(axis=1)
    feature_names = np.array(vectorizer.get_feature_names_out())
    df_a_2 = pd.concat([df_a, count_vect_df], axis=1, join='inner')
    mglearn.tools.visualize_coefficients(count_vect_df["TOTAL"], feature_names, n_top_features=20)


def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus


def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus


def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white',
                          stopwords=STOPWORDS,
                          width=3000,
                          height=3000,
                          max_words=200,
                          random_state=42
                          ).generate(str_corpus(corpus))
    return wordCloud


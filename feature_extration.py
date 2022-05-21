from pathlib import Path
import mglearn as mglearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud, STOPWORDS
from classification_algorithms import load_data


def bag_of_words_vizualizer():
    # загрузка данных из файла
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step2', 'train.csv')
    df_final_path = Path(work_path, 'dataset\\clear\\step3', 'train.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]

    # Представление через WordCloud
    # corpus = get_corpus(df_a['ABSTRACT'].values)
    #     procWordCloud = get_wordCloud(corpus)
    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(procWordCloud)
    #     plt.axis('off')
    #     plt.subplot(1, 2, 1)

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=10000, min_df=0.05, max_df=0.7)
    X_physics = vectorizer.fit_transform(df_a["ABSTRACT"])
    count_vect_df = pd.DataFrame(X_physics.todense(), columns=vectorizer.get_feature_names_out(), index=df_a.index)
    count_vect_df = count_vect_df.T
    count_vect_df["TOTAL"] = count_vect_df.sum(axis=1)
    feature_names = np.array(vectorizer.get_feature_names_out())
    df_a_2 = pd.concat([df_a, count_vect_df], axis=1, join='inner')
    dictionary=dict(zip(count_vect_df.index,count_vect_df["TOTAL"]))
    wordCloud = WordCloud(background_color='white',
                          stopwords=STOPWORDS,
                          width=2000,
                          height=2000,
                          max_words=500,
                          random_state=42
                          ).generate_from_frequencies(dictionary)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 1, 1)
    plt.imshow(wordCloud)
    plt.axis('off')
    plt.savefig('foo.png')

    #     plt.figure(figsize=(20, 10))
    #     plt.hist(count_vect_df["TOTAL"],20)
    #     a = plt.figure(figsize=(20, 20))
    #     bw=0.5
    #     plt.bar(count_vect_df["TOTAL"].nlargest(40).index, count_vect_df["TOTAL"].nlargest(40),bw,color="green")
    #     plt.xticks(ticks= range(len(count_vect_df["TOTAL"].nlargest(40).index)),labels= count_vect_df["TOTAL"].nlargest(40).index, rotation=90)
    #     plt.bar(count_vect_df["TOTAL"].nlargest(40).index, count_vect_df["TOTAL"].nlargest(40),bw,color="red")
    #     a = count_vect_df["TOTAL"].nlargest(40).plot("red")
    #     b = count_vect_df["TOTAL"].nlargest(40).plot("green")
    #     a.show()
    #     b.show()



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
                          width=2000,
                          height=2000,
                          max_words=200,
                          random_state=42
                          ).generate_from_frequencies()
    return wordCloud


def knn_vizualizer():
    mglearn.plots.plot_knn_classification(n_neighbors=3)

    # загрузка данных
    df_a = load_data()

    X, y = mglearn.datasets.make_forge()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 5], axes):
        # создаем объект-классификатор и подгоняем в одной строке
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:,0],X[:,1], ax=ax)
        ax.set_title("количество соседей:{}".format(n_neighbors))
        ax.set_xlabel("признак 0")
        ax.set_ylabel("признак 1")
    axes[0].legend(loc=3)

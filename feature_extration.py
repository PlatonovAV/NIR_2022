from pathlib import Path

import mglearn as mglearn
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud, STOPWORDS

from support_functions import load_data


def bag_of_words_vizualizer():
    # загрузка данных из файла
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step2', 'train.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]

    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10000, min_df=0.01, max_df=0.7)
    x_physics = vectorizer.fit_transform(df_a["ABSTRACT"])
    count_vect_df = pd.DataFrame(x_physics.todense(), columns=vectorizer.get_feature_names_out(), index=df_a.index)
    count_vect_df = count_vect_df.T
    count_vect_df["TOTAL"] = count_vect_df.sum(axis=1)
    feature_names = np.array(vectorizer.get_feature_names_out())
    # df_a_2 = pd.concat([df_a, count_vect_df], axis=1, join='inner')
    dictionary = dict(zip(count_vect_df.index, count_vect_df["TOTAL"]))
    word_cloud = WordCloud(background_color='white',
                          stopwords=STOPWORDS,
                          width=2000,
                          height=2000,
                          max_words=500,
                          random_state=42
                          ).generate_from_frequencies(dictionary)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 1, 1)
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.savefig('foo.png')

    mglearn.tools.visualize_coefficients(count_vect_df["TOTAL"], feature_names, n_top_features=20)


def feature_extraction_3_variant_of_clear_datasets():
    # загрузка данных из файла
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step2', 'train.csv')
    df_final_path_v1 = Path(work_path, 'dataset\\clear\\step3', 'train_v1.csv')
    df_final_path_v2 = Path(work_path, 'dataset\\clear\\step3', 'train_v2.csv')
    df_final_path_v3 = Path(work_path, 'dataset\\clear\\step3', 'train_v3.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]

    vectorizer1 = CountVectorizer(ngram_range=(1, 1), max_features=10000, min_df=0.05, max_df=0.7)
    X_df = vectorizer1.fit_transform(df_a["ABSTRACT"])
    count_vect_df_1 = pd.DataFrame(X_df.todense(), columns=vectorizer1.get_feature_names_out(), index=df_a.index)
    df_a_cleared_v1 = pd.concat([df_a, count_vect_df_1], axis=1, join='inner')

    vectorizer2 = CountVectorizer(ngram_range=(2, 2), max_features=10000, min_df=0.01, max_df=0.7)
    X_df = vectorizer2.fit_transform(df_a["ABSTRACT"])
    count_vect_df_2 = pd.DataFrame(X_df.todense(), columns=vectorizer2.get_feature_names_out(), index=df_a.index)
    df_a_cleared_v2 = pd.concat([df_a, count_vect_df_2], axis=1, join='inner')

    df_a_cleared_v3 = pd.concat([df_a, count_vect_df_1, count_vect_df_2], axis=1, join='inner')
    df_a_cleared_v3 = df_a_cleared_v3.fillna(0)

    df_a_cleared_v1.to_csv(df_final_path_v1)
    df_a_cleared_v2.to_csv(df_final_path_v2)
    df_a_cleared_v3.to_csv(df_final_path_v3)


def scaler_compare():
    # загрузка данных из файла
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step3', 'train_v1.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler_1 = StandardScaler()
    scaler_1 = scaler_1.fit_transform(
        pd.concat([X_train["model"], X_train["use"], X_train.iloc[:, [1]], X_train.iloc[:, [2]]], axis=1))
    scaler_1 = pd.DataFrame(scaler_1, columns=['o1', 'o2', 'o3', 'o4'])
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(5, 5))
    ax1.set_title('Before Scaling')
    sns.kdeplot(X_train['model'], ax=ax1, label='model')
    sns.kdeplot(X_train['use'], ax=ax1, label='use')
    sns.kdeplot(X_train.iloc[:, 0], ax=ax1, label=X_train.columns[0])
    sns.kdeplot(X_train.iloc[:, 1], ax=ax1, label=X_train.columns[1])
    ax1.legend()
    ax1.plot(label='model')

    ax2.set_title('After Standard Scaler')
    sns.kdeplot(scaler_1['o1'], ax=ax2, label='model')
    sns.kdeplot(scaler_1['o2'], ax=ax2, label='use')
    sns.kdeplot(scaler_1['o3'], ax=ax2, label=X_train.columns[0])
    sns.kdeplot(scaler_1['o4'], ax=ax2, label=X_train.columns[1])
    ax2.legend()
    plt.show()

    from sklearn.preprocessing import RobustScaler
    scaler_2 = RobustScaler()
    scaler_2 = scaler_2.fit_transform(
        pd.concat([X_train["model"], X_train["use"], X_train.iloc[:, [1]], X_train.iloc[:, [2]]], axis=1))
    scaler_2 = pd.DataFrame(scaler_2, columns=['o1', 'o2', 'o3', 'o4'])

    ax3.set_title('After Robust Scaler')
    sns.kdeplot(scaler_2['o1'], ax=ax3, label='model')
    sns.kdeplot(scaler_2['o2'], ax=ax3, label='use')
    sns.kdeplot(scaler_2['o3'], ax=ax3, label=X_train.columns[0])
    sns.kdeplot(scaler_2['o4'], ax=ax3, label=X_train.columns[1])
    ax3.legend()
    ax1.set_xlabel('value')
    ax2.set_xlabel('value')
    ax3.set_xlabel('value')

    plt.show()


def extr():
    # загрузка данных из файла
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step3', 'train_v1.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]
    fig1 = dict.fromkeys(list(df_a.columns)[1:])
    for n in df_a.columns[2:20]:
        fig1[n] = plt.subplots(1)
        sns.set_theme(style="whitegrid")
        sns.boxplot(df_a[n])
        plt.show()
        fig1[n] = plt.subplots(1)
        fig1[n] = df_a[n].hist(figsize=(10, 10)).get_figure()
        # отсечение выбросов
        r = df_a[n].quantile(0.75) - df_a[n].quantile(0.25)
        maximum = df_a[n].quantile(0.75) + (1.5 * r)
        minimum = df_a[n].quantile(0.25) - (1.5 * r)
        plt.axvline(x=maximum, color='red')
        plt.axvline(x=minimum, color='red')
        plt.xlabel(n)
        plt.ylabel('Частота')


def str_corpus(corpus):
    corp = ''
    for i in corpus:
        corp += ' ' + i
    corp = corp.strip()
    return corp


def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus


def get_word_cloud(corpus):
    word_cloud = WordCloud(background_color='white',
                           stopwords=STOPWORDS,
                           width=2000,
                           height=2000,
                           max_words=200,
                           random_state=42
                           ).generate_from_frequencies(corpus)
    return word_cloud


def knn_vizualizer(file_name):
    mglearn.plots.plot_knn_classification(n_neighbors=3)

    # загрузка данных
    df_a = load_data(file_name)
    x, y = mglearn.datasets.make_forge()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 5], axes):
        # создаем объект-классификатор и подгоняем в одной строке
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x, y)
        mglearn.plots.plot_2d_separator(clf, x, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(x[:, 0], x[:, 1], ax=ax)
        ax.set_title("количество соседей:{}".format(n_neighbors))
        ax.set_xlabel("признак 0")
        ax.set_ylabel("признак 1")
    axes[0].legend(loc=3)

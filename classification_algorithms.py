from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data():
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step2', 'train.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]
    return df_a


def k_neighbors():
    # загрузка данных
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    scaler = StandardScaler()
    scaler.fit(X_train)

    print("Классификация без использования скалирования")
    classifier = KNeighborsClassifier(n_neighbors=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=100)
    classifier.fit(X_train_scaler, y_train)
    y_pred = classifier.predict(X_test_scaler)
    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Поиск оптимальных значений")
    knn = KNeighborsClassifier(algorithm="kd_tree")
    k_range = list(range(1, 106, 10))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc', verbose=3, return_train_score=True, n_jobs=-1)
    grid_search = grid.fit(X_train, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    plt.figure(figsize=(10, 10)).clf()
    for n in [1,2,5,10,25]:
        classifier = KNeighborsClassifier(n_neighbors=n,algorithm="kd_tree")
        classifier.fit(X_train_scaler, y_train)
        y_pred = classifier.predict(X_test_scaler)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))
        print('\n')

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = classifier.predict_proba(X_test)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='n= %s ' % n, color=col)
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()


def random_forest():
    # загрузка данных
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    forest = RandomForestClassifier(random_state=42, max_features=200, n_estimators=10, n_jobs=4)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    forest = RandomForestClassifier(random_state=42, max_features=200, n_estimators=10, n_jobs=4)
    forest.fit(X_train_scaler, y_train)
    y_pred = forest.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Поиск оптимальных значений")
    rnd_frst = RandomForestClassifier(random_state=42, max_features="sqrt")
    k_range = list([5, 10, 50, 100])
    param_grid = dict(n_estimators=k_range)
    grid = GridSearchCV(rnd_frst, param_grid, cv=5, scoring='roc_auc', verbose=3, return_train_score=True, n_jobs=-1)
    grid_search = grid.fit(X_train, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    plt.figure(figsize=(10, 10)).clf()
    for n in k_range:
        classifier = RandomForestClassifier(n_estimators=n, random_state=42, max_features="sqrt", n_jobs=4)
        classifier.fit(X_train_scaler, y_train)
        y_pred = classifier.predict(X_test_scaler)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))
        print('\n')

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = classifier.predict_proba(X_test_scaler)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='n= %s ' % n, color=col)
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()


def decision_tree():
    # загрузка данных
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    dtree = DecisionTreeClassifier(random_state=42, max_features=200)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    dtree = DecisionTreeClassifier(random_state=42, max_features=200)
    dtree.fit(X_train_scaler, y_train)
    y_pred = dtree.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()


def naive_bayes_bernoulli():
    # загрузка данных
    print("Классификация с помощью Наивного Байеса Бернулли\n")
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    nb = BernoulliNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    nb = BernoulliNB()
    nb.fit(X_train_scaler, y_train)
    y_pred = nb.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()


def naive_bayes_gaussian():
    print("Классификация с помощью Наивного Байеса Гаусса\n")
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    nb = GaussianNB()
    nb.fit(X_train_scaler, y_train)
    y_pred = nb.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()


def nb_compare():
    print("сравнение методов:")
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=2000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    plt.figure(figsize=(10, 10)).clf()
    print()
    nbb = BernoulliNB()
    nbg = GaussianNB()

    print("Метод Бернулли")
    nbb.fit(X_train_scaler, y_train)
    y_pred = nbb.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    col = (np.random.random(), np.random.random(), np.random.random())
    Roc_data = nbb.predict_proba(X_test_scaler)
    fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
    plt.plot(fpr_roc, tpr_roc, label='BernoulliNB', color=col)
    plt.plot((0.0, 1.0), (0.0, 1.0))
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.legend()

    print("Метод Гаусса")
    nbg.fit(X_train_scaler, y_train)
    y_pred = nbg.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()
    col = (np.random.random(), np.random.random(), np.random.random())
    Roc_data = nbg.predict_proba(X_test_scaler)
    fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
    plt.plot(fpr_roc, tpr_roc, label='GaussianNB', color=col)
    plt.plot((0.0, 1.0), (0.0, 1.0))
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.legend()


def bagging():
    # загрузка данных
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=10, random_state=42, max_features=200,
                            n_jobs=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=10, random_state=42,
                            n_jobs=4, max_features=200)
    clf.fit(X_train_scaler, y_train)
    y_pred = clf.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Поиск оптимальных значений")
    clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), random_state=42,
                            max_features=200)
    k_range = list([5, 10, 50, 100])
    param_grid = dict(n_estimators=k_range)
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', verbose=3, return_train_score=True, n_jobs=5)
    grid_search = grid.fit(X_train, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    plt.figure(figsize=(10, 10)).clf()
    for n in k_range:
        clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=n, random_state=42,
                                n_jobs=4, max_features=200)
        clf.fit(X_train_scaler, y_train)
        y_pred = clf.predict(X_test_scaler)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))
        print('\n')

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = clf.predict_proba(X_test_scaler)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='n= %s ' % n, color=col)
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()


def ada_boost():
    # загрузка данных
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train_scaler, y_train)
    y_pred = clf.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Поиск оптимальных значений")
    clf = AdaBoostClassifier(random_state=42)
    k_range = list([5, 10, 50, 100])
    param_grid = dict(n_estimators=k_range)
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', verbose=3, return_train_score=True, n_jobs=5)
    grid_search = grid.fit(X_train, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    plt.figure(figsize=(10, 10)).clf()
    for n in k_range:
        clf = AdaBoostClassifier(n_estimators=n, random_state=42)
        clf.fit(X_train_scaler, y_train)
        y_pred = clf.predict(X_test_scaler)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))
        print('\n')

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = clf.predict_proba(X_test_scaler)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='n= %s ' % n, color=col)
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()


def gradient_boost():
    # загрузка данных
    df_a = load_data()

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.7, max_features=1000)
    X_df = vectorizer.fit_transform(df_a["ABSTRACT"]).toarray()

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(X_df, df_a['LABEL'], train_size=0.75, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    clf = GradientBoostingClassifier(n_estimators=10, random_state=42, learning_rate=0.01, max_depth=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    clf = GradientBoostingClassifier(n_estimators=10, random_state=42, learning_rate=0.01, max_depth=2)
    clf.fit(X_train_scaler, y_train)
    y_pred = clf.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Поиск оптимальных значений")
    clf = GradientBoostingClassifier(random_state=42)
    k_range = list([2, 10, 50, 100])
    kk_range = list([0.1, 0.5, 1.0])
    kkk_range = list([2, 4, 6])
    param_grid = dict(n_estimators=k_range, learning_rate=kk_range, max_depth=kkk_range)
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', verbose=3, return_train_score=True, n_jobs=-1)
    grid_search = grid.fit(X_train, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    plt.figure(figsize=(10, 10)).clf()
    for n in [5, 10 ,50 ,100]:
        for m in [0.1, 0.5, 1.0]:
            for k in [2,4,6]:
                clf = GradientBoostingClassifier(n_estimators=n, random_state=42, learning_rate=m, max_depth=k)
                clf.fit(X_train_scaler, y_train)
                y_pred = clf.predict(X_test_scaler)
                quality = confusion_matrix(y_test, y_pred)
                print('параметры:\nn_estimators=', n, '\nlearning_rate=', m, '\nmax_depth=', k)
                print('полнота', quality[0, 0] / sum(quality[0, :]))
                print('специфичность', quality[1, 1] / sum(quality[1, :]))
                print('\n')

                col = (np.random.random(), np.random.random(), np.random.random())
                Roc_data = clf.predict_proba(X_test_scaler)
                fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
                plt.plot(fpr_roc, tpr_roc, label='n= {},m= {}, k={}'.format(n,m,k), color=col)
                plt.plot((0.0, 1.0), (0.0, 1.0))
                plt.xlabel('True Positive Rate')
                plt.ylabel('False Positive Rate')
                plt.legend()

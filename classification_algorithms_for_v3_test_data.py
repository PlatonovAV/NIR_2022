import timeit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier

from support_functions import load_data, recall_specificity_scoring

MAX_FEATURES = None
N_JOBS = -1


def k_neighbors(txt):
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    s = timeit.default_timer()
    print("Классификация без использования скалирования")
    classifier = KNeighborsClassifier(n_jobs=N_JOBS)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))
    e = timeit.default_timer()
    print("elapsed time:", e - s)

    print()

    s = timeit.default_timer()
    print("Классификация с использованием скалирования")
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_jobs=N_JOBS)
    classifier.fit(X_train_scaler, y_train)
    y_pred = classifier.predict(X_test_scaler)
    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))
    e = timeit.default_timer()
    print("elapsed time:", e - s)
    # from sklearn.metrics import plot_confusion_matrix
    # plot_confusion_matrix(classifier, X_test_scaler, y_test)
    # plt.show()
    print()

    clf = KNeighborsClassifier(n_neighbors=300, algorithm='kd_tree', n_jobs=-1)

    a_scaler = RobustScaler()
    recall_specificity_scoring(df_a, clf, a_scaler)

    print("Поиск оптимальных значений")
    knn = KNeighborsClassifier(algorithm="ball_tree")
    a_scaler = RobustScaler()
    k_range = [10, 20, 50, 100, 150, 200, 250, 300, 400]
    k_alg = ['ball_tree', 'kd_tree', 'brute']
    param_grid = dict(n_neighbors=k_range, algorithm=k_alg)
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='balanced_accuracy', verbose=4, return_train_score=True,
                        n_jobs=N_JOBS)
    grid_search = grid.fit(a_scaler.fit_transform(df_a.iloc[:, list(range(2, len(df_a.columns)))]), df_a['LABEL'])
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    plt.figure(figsize=(10, 10)).clf()
    for n in k_range:
        classifier = KNeighborsClassifier(n_neighbors=n, algorithm="kd_tree", n_jobs=N_JOBS)
        classifier.fit(X_train_scaler, y_train)
        y_pred = classifier.predict(X_test_scaler)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))
        print('\n')

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = classifier.predict_proba(X_test.values)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='n = %s' % n, color=col)
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()

    # вычисление времени выполнения
    clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=150)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    start_time = timeit.default_timer()
    for n in range(3):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('время выполнения:', end_time - start_time)


def random_forest():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)
    scaler = RobustScaler()
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
    s = timeit.default_timer()
    print("Классификация без использования скалирования")
    forest = RandomForestClassifier(random_state=42, n_estimators=10, n_jobs=N_JOBS)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))
    e = timeit.default_timer()
    print("elapsed time:", e - s)

    print()
    s = timeit.default_timer()
    print("Классификация с использованием скалирования")
    forest = RandomForestClassifier(random_state=42, n_estimators=10, n_jobs=N_JOBS)
    forest.fit(X_train_scaler, y_train)
    y_pred = forest.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))
    e = timeit.default_timer()
    print("elapsed time:", e - s)
    print()

    print("Поиск оптимальных значений")
    rnd_frst = RandomForestClassifier(random_state=42)
    k_range = [2, 5, 10, 20, 50, 100]
    param_grid = dict(n_estimators=k_range)
    a_scaler = RobustScaler()
    grid = GridSearchCV(rnd_frst, param_grid, cv=3, scoring='roc_auc', verbose=3, return_train_score=True,
                        n_jobs=N_JOBS)
    grid_search = grid.fit(a_scaler.fit_transform(df_a.iloc[:, list(range(2, len(df_a.columns)))]), df_a['LABEL'])
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    plt.figure(figsize=(10, 10)).clf()
    for n in k_range:
        classifier = RandomForestClassifier(n_estimators=n, random_state=42, max_features=MAX_FEATURES, n_jobs=N_JOBS)
        classifier.fit(X_train_scaler, y_train)
        y_pred = classifier.predict(X_test_scaler)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))
        print('\n')

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = classifier.predict_proba(X_test_scaler)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='ближайших соседей = %s ' % n, color=col)
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()


def decision_tree():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)
    scaler = RobustScaler()
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
    dtree = DecisionTreeClassifier(random_state=42, max_features=MAX_FEATURES)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    cn = ['Computer Science', 'Physics']
    dtree = DecisionTreeClassifier(random_state=42, max_depth=5, max_features=MAX_FEATURES)
    dtree.fit(X_train_scaler, y_train)
    y_pred = dtree.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    a_scaler = RobustScaler()
    clf = DecisionTreeClassifier(random_state=42, max_features=MAX_FEATURES)
    recall_specificity_scoring(df_a, clf, a_scaler)

    clf = DecisionTreeClassifier(random_state=42)

    k_criterion = ['gini', 'entropy', 'log_loss']
    k_max_depth = [1, 2, 5, 10, 12, 15, 20, 30, 50, 100, 150, 200, None]
    k_min_samples_split = [2, 5, 10, 12, 15, 20, 30, 50, 100, 150, 200, 300, 500, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    k_min_samples_leaf = [1, 2, 5, 10, 12, 15, 20, 30, 50, 100, 150, 200, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5,
                          2.0, 3.0]
    param_grid = dict(criterion=k_criterion, max_depth=k_max_depth, min_samples_split=k_min_samples_split
                      , min_samples_leaf=k_min_samples_leaf)
    time_start = timeit.default_timer()
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', verbose=1,
                        return_train_score=True, n_jobs=N_JOBS)
    grid_search = grid.fit(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'])
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))
    time_stop = timeit.default_timer()
    clf = DecisionTreeClassifier(criterion=grid_search.best_params_['criterion'],
                                 max_depth=grid_search.best_params_['max_depth'],
                                 min_samples_split=grid_search.best_params_['min_samples_split'],
                                 min_samples_leaf=grid_search.best_params_['min_samples_leaf'])

    a_scaler = RobustScaler()
    recall_specificity_scoring(df_a, clf, a_scaler)
    print("Время выполнения: ", time_stop - time_start, "\nВремя в минутах: ", (time_stop - time_start) / 60)

    # roc_auc {'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 12, 'min_samples_split': 200}
    # Accuracy for our training dataset with tuning is : 93.52%

    # Полнота:  0.9007862242240403
    # Специфичность:  0.8349609375

    # balanced {'criterion': 'entropy', 'max_depth': 100, 'min_samples_leaf': 5, 'min_samples_split': 50}
    # 87 %
    # Полнота:  0.8755312477658282
    #  Специфичность:  0.8681640625

    # вычисление времени выполнения
    clf = DecisionTreeClassifier(max_depth=30, min_samples_split=12, min_samples_leaf=200)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('время выполнения:', end_time - start_time)


def naive_bayes_bernoulli():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)
    scaler = RobustScaler()
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

    clf = BernoulliNB()
    a_scaler = RobustScaler()
    recall_specificity_scoring(df_a, clf, a_scaler)

    print("Поиск оптимальных значений")
    nb = BernoulliNB()
    k_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 3.0]
    kk_range = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5]
    param_grid = dict(alpha=k_range, binarize=kk_range)

    grid = GridSearchCV(nb, param_grid, cv=5, scoring='roc_auc', verbose=4, return_train_score=True, n_jobs=N_JOBS)
    grid_search = grid.fit(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'])
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    clf = BernoulliNB(alpha=grid_search.best_params_['alpha'], binarize=grid_search.best_params_['binarize'])
    a_scaler = RobustScaler()
    recall_specificity_scoring(df_a, clf, a_scaler)

    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('время выполнения:', end_time - start_time)


def naive_bayes_multinomial():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)
    scaler = RobustScaler()
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
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    clf = MultinomialNB()
    recall_specificity_scoring(df_a, clf)

    print("Поиск оптимальных значений")
    nb = MultinomialNB()
    k_range = list(np.arange(0.005, 3.0, 0.005))
    param_grid = dict(alpha=k_range)
    grid = GridSearchCV(nb, param_grid, cv=5, scoring='roc_auc', verbose=4, return_train_score=True, n_jobs=N_JOBS)
    grid_search = grid.fit(X_train, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    clf = MultinomialNB(alpha=grid_search.best_params_['alpha'])
    recall_specificity_scoring(df_a, clf)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('время выполнения:', end_time - start_time)


def nb_compare():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    plt.figure(figsize=(10, 10)).clf()
    print()
    nbb = BernoulliNB()
    nbm = MultinomialNB()

    print("Метод Бернулли")
    nbb.fit(X_train, y_train)
    y_pred = nbb.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    col = (np.random.random(), np.random.random(), np.random.random())
    Roc_data = nbb.predict_proba(X_test)
    fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
    plt.plot(fpr_roc, tpr_roc, label='Наивный Байес Бернулли', color="green")
    plt.plot((0.0, 1.0), (0.0, 1.0))
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.legend()

    print("Метод Гаусса")
    nbm.fit(X_train, y_train)
    y_pred = nbm.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()
    col = (np.random.random(), np.random.random(), np.random.random())
    Roc_data = nbm.predict_proba(X_test)
    fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
    plt.plot(fpr_roc, tpr_roc, label='Полиномиальный Наивный Байкс', color="red")
    plt.plot((0.0, 1.0), (0.0, 1.0))
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.legend()


def bagging():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    base = KNeighborsClassifier()
    base_classifier = base

    print()
    start_time = timeit.default_timer()
    # без скалирования
    print("Классификация без использования скалирования")
    clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42, n_jobs=N_JOBS)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")
    base_classifier = base
    clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42, n_jobs=N_JOBS)
    clf.fit(X_train_scaler, y_train)
    y_pred = clf.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))
    end_time = timeit.default_timer()
    print('время выполнения: ', end_time - start_time)
    print()

    print("Поиск оптимальных значений")
    base_classifier = base
    clf = BaggingClassifier(base_estimator=base_classifier, random_state=42, n_jobs=N_JOBS)
    k_range = [2, 5, 10, 20, 50]
    k_max_samples = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    param_grid = dict(n_estimators=k_range, max_samples=k_max_samples)
    grid = GridSearchCV(clf, param_grid, scoring='balanced_accuracy', verbose=3, return_train_score=True, n_jobs=2)
    grid_search = grid.fit(X_train_scaler, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    a_scaler = RobustScaler()
    clf = BaggingClassifier(base_estimator=base_classifier, random_state=42,
                            n_estimators=grid_search.best_params_['n_estimators'],
                            max_samples=grid_search.best_params_['max_samples']
                            , n_jobs=N_JOBS)
    recall_specificity_scoring(df_a, a_scaler, clf)

    # метод случайных соседей, max_samples должно быть больше чем n_neighbors
    # Оптимальные параметры:
    # max_samples: 1.0 (0.2)
    # n_estimators: 10 (50)
    # 87 % (95)
    # Оценка:
    # Полнота:0.87 (0.65)
    # Специфичность:0.88 (0.95)
    # *********************
    # метод случайного леса без параметров()
    # Оптимальные параметры:
    # 'max_samples': 1.0, 'n_estimators': 20 ()
    # Оценка 88 % (94)
    # Полнота:0.89 (0.89)
    # Специфичность:0.89 (0.87)
    # метод случайного леса c параметрами определёнными в первый раз
    # Оптимальные параметры:
    # {'max_samples': 1.0, 'n_estimators': 10} ({'max_samples': 1.0, 'n_estimators': 10})
    # Оценка 80 (88)
    # Полнота: 0.81 (0.82)
    # Специфичность: 0.81 (0.80)
    # *********************
    # Полиномиальный Байес
    # Оптимальные параметры: {'max_samples': 0.02, 'n_estimators': 50} ({'max_samples': 0.02, 'n_estimators': 50})
    # Оценка ()
    # Полнота:0.92 (0.92)
    # Специфичность:0.88 (0.88)
    # *********************
    # Байес Бернулли
    # Оптимальные параметры:
    # {'max_samples': 1.0, 'n_estimators': 10} ({'max_samples': 0.5, 'n_estimators': 20})
    # Оценка: 97
    # Полнота: 0.91 0.91
    # Специфичность: 0.91 0.91
    # *********************

    k_range = [1, 2, 5, 10, 20, 50]

    base = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=150)
    plt.figure(figsize=(10, 10)).clf()
    base_classifier = base
    for n in k_range:
        clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, max_samples=1.0, random_state=42,
                                n_jobs=N_JOBS)
        clf.fit(X_train_scaler, y_train)
        y_pred = clf.predict(X_test_scaler)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = clf.predict_proba(X_test_scaler)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='Второй вариант оптимальных значений', color=col)
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()
        print("n = ", n)
        print('\n')

    # ********************************************
    # Время бэггинга для K-ближайших соседй
    base_clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=150)
    clf = BaggingClassifier(base_estimator=base_clf, n_estimators=50, max_samples=0.2, n_jobs=-1)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    start_time = timeit.default_timer()
    for n in range(10):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('Время выполнения', end_time - start_time)

    # ********************************************
    # Время бэггинга для Наивног Байеса Бернулли
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.fit_transform(X_test)
    base_clf = BernoulliNB()
    clf = BaggingClassifier(base_estimator=base_clf, n_estimators=20, max_samples=0.5, n_jobs=-1)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('Время выполнения', end_time - start_time)

    # ********************************************
    # Время бэггинга для полиномиального Наивного Байеса
    base_clf = MultinomialNB()
    clf = BaggingClassifier(base_estimator=base_clf, n_estimators=50, max_samples=0.02, n_jobs=-1)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('Время выполнения', end_time - start_time)

    # ********************************************
    # Время бэггинга для дерева решений
    base_clf = DecisionTreeClassifier(max_depth=30, min_samples_split=12, min_samples_leaf=200)
    clf = BaggingClassifier(base_estimator=base_clf, n_estimators=50, max_samples=1.0, n_jobs=-1)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('Время выполнения', end_time - start_time)


def ada_boost():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    base_classifier = MultinomialNB()

    print('количество объектов в тренировочной выборке:\n', y_train.value_counts(), "\n")
    print('количество объектов в тестовой выборке: \n', y_test.value_counts(), "\n")

    print("Процентное соотношение выборок:")
    print(y_train.value_counts()[0] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')
    print(y_train.value_counts()[1] / (y_train.value_counts()[0] + y_train.value_counts()[1]) * 100, ' %')

    print()
    # без скалирования
    print("Классификация без использования скалирования")
    clf = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    clf = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42)
    clf.fit(X_train_scaler, y_train)
    y_pred = clf.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    clf = AdaBoostClassifier(n_estimators=10, random_state=42)

    recall_specificity_scoring(df_a, clf)

    print()

    print("Поиск оптимальных значений")
    clf = AdaBoostClassifier(base_estimator=base_classifier, random_state=42)
    k_range = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100]
    k_learning_rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0]
    k_algorithm = ['SAMME', 'SAMME.R']
    param_grid = dict(n_estimators=k_range, learning_rate=k_learning_rate, algorithm=k_algorithm)
    grid = GridSearchCV(clf, param_grid, scoring='balanced_accuracy', verbose=3, return_train_score=True, n_jobs=-1)
    grid_search = grid.fit(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'])
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    clf = AdaBoostClassifier(base_estimator=base_classifier, random_state=42,
                             n_estimators=grid_search.best_params_['n_estimators'],
                             learning_rate=grid_search.best_params_['learning_rate'],
                             algorithm=grid_search.best_params_['algorithm'])

    recall_specificity_scoring(df_a, clf)

    # Результаты работы алогритмов
    # (в скобках указаны значения полученные при оценке 'roc_auc')
    # Деревья решений при максимальной глубине равной 1
    # Параметры
    # {'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'n_estimators': 200} ({'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'n_estimators': 200})
    # Оценка
    # Проценты: (97,5) 0,92
    # Полнота: (0.93) 0,93
    # Специфичность: (0.91) 0,91
    # ***************
    # Деревья решений с определенными ранее параметрами
    # Параметры
    #  {'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'n_estimators': 40} ({'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 200})
    # Оценка
    # Проценты: (97)
    # Полнота: 0.92 (0.93)
    # Специфичность: 0.91 (0.92)
    # ***************
    # Полиномиальный Наивный Байес
    # Параметры
    # {'algorithm': 'SAMME.R', 'learning_rate': 0.2, 'n_estimators': 100} ({'algorithm': 'SAMME.R', 'learning_rate': 0.4, 'n_estimators': 50})
    # Оценка для roc_auc аналогично
    # Проценты: 92
    # Полнота: 0.91 (0,94)
    # Специфичность: 0.93 (0,83)
    # ***************
    # Наивный Байес Бернулли
    # Параметры
    # {'algorithm': 'SAMME', 'learning_rate': 0.05, 'n_estimators': 100} ({'algorithm': 'SAMME', 'learning_rate': 0.05, 'n_estimators': 100})
    # Оценка
    # Проценты:
    # Полнота: 0,94 (94)
    # Специфичность:0,88 (88)

    plt.figure(figsize=(10, 10)).clf()
    for n in k_range:
        clf = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=n, random_state=42,
                                 algorithm='SAMME.R',
                                 learning_rate=0.2)
        clf = base_classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        quality = confusion_matrix(y_test, y_pred)
        print('полнота', quality[0, 0] / sum(quality[0, :]))
        print('специфичность', quality[1, 1] / sum(quality[1, :]))
        print('\n')

        col = (np.random.random(), np.random.random(), np.random.random())
        Roc_data = clf.predict_proba(X_test)
        fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
        plt.plot(fpr_roc, tpr_roc, label='полиномиальный Байес Бернулли', color='green')
        plt.plot((0.0, 1.0), (0.0, 1.0))
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.legend()
        plt.show()

    # ********************************************
    # Время бустинга для Наивног Байеса Бернулли
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.fit_transform(X_test)
    base_clf = BernoulliNB()
    clf = AdaBoostClassifier(base_estimator=base_clf, algorithm='SAMME', learning_rate=0.05, n_estimators=50)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('Время выполнения', end_time - start_time)

    # ********************************************
    # Время бустинга для полиномиального Наивного Байеса
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.fit_transform(X_test)
    base_clf = MultinomialNB()
    clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=50, algorithm='SAMME', learning_rate=0.05)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('Время выполнения', end_time - start_time)

    # ********************************************
    # Время бустинга для дерева решений
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.fit_transform(X_test)
    base_clf = DecisionTreeClassifier(max_depth=30, min_samples_split=12, min_samples_leaf=200)
    clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=40, algorithm='SAMME.R', learning_rate=0.5)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for n in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print('Время выполнения', end_time - start_time)


def gradient_boost():
    # загрузка данных
    df_a = load_data('train_v1.csv')

    # формирование тестовых выборок
    X_train, X_test, y_train, y_test = train_test_split(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL'],
                                                        train_size=0.75, random_state=42)
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
    clf = GradientBoostingClassifier(n_estimators=10, random_state=42, learning_rate=0.01, max_depth=4,
                                     max_features=MAX_FEATURES)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Классификация с использованием скалирования")

    clf = GradientBoostingClassifier(n_estimators=10, random_state=42, learning_rate=0.01, max_depth=2,
                                     max_features=MAX_FEATURES)
    clf.fit(X_train_scaler, y_train)
    y_pred = clf.predict(X_test_scaler)

    quality = confusion_matrix(y_test, y_pred)
    print('полнота', quality[0, 0] / sum(quality[0, :]))
    print('специфичность', quality[1, 1] / sum(quality[1, :]))

    print()

    print("Поиск оптимальных значений")
    clf = GradientBoostingClassifier(random_state=42)
    k_range = [2, 5, 10, 25, 50, 75, 100]
    k_learning_rate = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    k_max_depth = [2, 4, 8, 16]
    k_loss = ['log_loss', 'exponential']
    k_criterion = ['friedman_mse', 'squared_error']
    k_subsample = [0.25, 0.5, 0.75, 1.0]

    param_grid = dict(n_estimators=k_range, learning_rate=k_learning_rate, max_depth=k_max_depth, loss=k_loss,
                      criterion=k_criterion, subsample=k_subsample)
    grid = GridSearchCV(clf, param_grid, scoring='balanced_accuracy', cv=4, verbose=3, return_train_score=True,
                        n_jobs=-1)
    grid_search = grid.fit(scaler.fit_transform(df_a.iloc[:, list(range(2, len(df_a.columns)))]), df_a['LABEL'])
    print(grid_search)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

    a_scaler = RobustScaler()
    clf = GradientBoostingClassifier(random_state=42,
                                     n_estimators=grid_search.best_params_['n_estimators'],
                                     learning_rate=grid_search.best_params_['learning_rate'],
                                     max_depth=grid_search.best_params_['max_depth'],
                                     loss=grid_search.best_params_['loss'],
                                     criterion=grid_search.best_params_['criterion'],
                                     subsample=grid_search.best_params_['subsample'])
    recall_specificity_scoring(df_a, clf,a_scaler)
    # {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'exponential', 'max_depth': 16, 'n_estimators': 100, 'subsample': 0.5}
    # ******
    #  Полнота:  0.9347384869932899
    #  Специфичность:  0.914453125
    # *****

    plt.figure(figsize=(10, 10)).clf()
    for n in [5, 10, 50, 100]:
        for m in [0.1, 0.5, 1.0]:
            for k in [2, 4, 6]:
                clf = GradientBoostingClassifier(n_estimators=n, random_state=42, learning_rate=m, max_depth=k,
                                                 max_features=MAX_FEATURES)

                clf = DecisionTreeClassifier(max_depth=30, min_samples_leaf=200, min_samples_split=12)
                clf.fit(X_train_scaler, y_train)
                y_pred = clf.predict(X_test_scaler)
                quality = confusion_matrix(y_test, y_pred)
                print('полнота', quality[0, 0] / sum(quality[0, :]))
                print('специфичность', quality[1, 1] / sum(quality[1, :]))
                print('\n')

                col = (np.random.random(), np.random.random(), np.random.random())
                Roc_data = clf.predict_proba(X_test_scaler)
                fpr_roc, tpr_roc, threshold_roc = roc_curve(y_test, Roc_data[:, 1], pos_label='Physics')
                plt.plot(fpr_roc, tpr_roc, label='Дерево решений', color='blue')
                plt.plot((0.0, 1.0), (0.0, 1.0))
                plt.xlabel('True Positive Rate')
                plt.ylabel('False Positive Rate')
                plt.legend()
                plt.show()

    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.fit_transform(X_test)
    clf = GradientBoostingClassifier(random_state=42,
                                     n_estimators=50,
                                     learning_rate=0.1,
                                     max_depth=16,
                                     loss='exponential',
                                     criterion='friedman_mse',
                                     subsample=0.5)
    clf.fit(X_train, y_train)
    X_test_scaler_2 = X_test.head(2500)
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    X_test_scaler_2 = pd.concat([X_test_scaler_2, X_test_scaler_2])
    start_time = timeit.default_timer()
    for i in range(20):
        y_pred = clf.predict(X_test_scaler_2)
    end_time = timeit.default_timer()
    print("время выполнения", end_time - start_time)

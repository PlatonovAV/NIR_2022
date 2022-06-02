from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate


def recall_specificity_scoring(df_a, clf, scaler=None):
    if scaler is not None:
        cv_results = cross_validate(clf.fit(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL']),
                                    scaler.fit_transform(df_a.iloc[:, list(range(2, len(df_a.columns)))]),
                                    df_a["LABEL"],
                                    scoring=confusion_matrix_scorer)
    else:
        cv_results = cross_validate(clf.fit(df_a.iloc[:, list(range(2, len(df_a.columns)))], df_a['LABEL']),
                                    (df_a.iloc[:, list(range(2, len(df_a.columns)))]), df_a["LABEL"],
                                    scoring=confusion_matrix_scorer)

    recal = 0
    specificit = 0
    for n in range(len(cv_results['test_tp'])):
        recal += cv_results['test_tp'][n] / (cv_results['test_tp'][n] + cv_results['test_fn'][n])
        specificit += cv_results['test_tn'][n] / (cv_results['test_tn'][n] + cv_results['test_fp'][n])
        print('полнота', cv_results['test_tp'][n] / (cv_results['test_tp'][n] + cv_results['test_fn'][n]))
        print('специфичность', cv_results['test_tn'][n] / (cv_results['test_tn'][n] + cv_results['test_fp'][n]))
        print()
    recal = recal / len(cv_results['test_tp'])
    specificit = specificit / len(cv_results['test_tp'])
    print("******\n", "Полнота: ", recal, '\n', 'Специфичность: ', specificit, '\n*****\n')
    print(cv_results)


def load_data(txt):
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step3', txt)
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)
    df_a.index.names = ["ID"]
    return df_a


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tp': cm[0, 0], 'fn': cm[0, 1], 'fp': cm[1, 0], 'tn': cm[1, 1]}

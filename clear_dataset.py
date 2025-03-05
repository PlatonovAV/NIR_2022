# используемые библиотеки
from pathlib import Path

import nltk
import pandas as pd
import spacy  # need to download package: in cmd "python -m spacy download en_core_web_lg"
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


# Предназначен для очистки текста
def clear_data():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\raw', 'train.csv')
    df_final_path = Path(work_path, 'dataset\\clear\\step1', 'train.csv')

    # импорт данных
    print("Процесс импорта данных начат")
    df_a = pd.read_csv(df_a_path, sep=',')
    print("Процесс импорта данных завершён")

    # Удаление ненужных столбцов
    print("Процесс удаления ненужных столбцов начат")
    df_a.drop(['ID'], axis=1, inplace=True)
    df_a.drop(['TITLE'], axis=1, inplace=True)
    print("Процесс удаления ненужных столбцов завершён")

    # удаление переносов (изначально в текстах есть переносы)
    print("Процесс удаления переносов начат")
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: " ".join(x.splitlines()))
    print("Процесс удаления переносов завершён")

    # приведение к ниженму регистру
    print("Процесс приведения к нижнему регистру начат")
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: x.lower())
    print("Процесс приведения к нижнему регистру завершён")

    # Удаление пунктуациии из текста
    print("Процесс удаления пунктуации начат")
    tokenizer = RegexpTokenizer(r'\w+')
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: tokenizer.tokenize(x))
    print("Процесс удаления пунктуации завершён")

    # удаление стопслов
    print("Процесс удаления стоп-слов начат")
    stop_words = set(stopwords.words("english"))
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(
        lambda x: [word for word in x if word not in stop_words and len(word) > 1 and not word.isdigit()])
    print("Процесс удаления стоп-слов завершён")

    # лемматизация слов
    print("Процесс лемматизации слов начат")
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: " ".join(x))
    nlp = spacy.load("en_core_web_lg")
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))
    print("Процесс лемматизации слов завершён")

    # сохранение результатов
    df_a.to_csv(df_final_path)

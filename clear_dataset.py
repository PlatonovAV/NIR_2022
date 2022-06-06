# используемые библиотеки
from pathlib import Path
import nltk
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import spacy  # need to download package: in cmd "python -m spacy download en_core_web_lg"


# Предназначен для очистки текста
def clear_data():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    work_path = Path.cwd()
    df_a_path = Path(work_path, 'dataset\\raw', 'train.csv')
    df_final_path = Path(work_path, 'dataset\\clear\\step1', 'train.csv')

    # импорт данных
    df_a = pd.read_csv(df_a_path, sep=',')

    # Удаление ненужных столбцов
    df_a.drop(['ID'], axis=1, inplace=True)
    df_a.drop(['TITLE'], axis=1, inplace=True)

    # удаление переносов (изначально в текстах есть переносы)
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: " ".join(x.splitlines()))

    # приведение к ниженму регистру
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: x.lower())

    # Удаление пунктуациии из текста
    tokenizer = RegexpTokenizer(r'\w+')
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: tokenizer.tokenize(x))

    # удаление стопслов
    stop_words = set(stopwords.words("english"))
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(
        lambda x: [word for word in x if word not in stop_words and len(word) > 1 and not word.isdigit()])

    # лемматизация слов
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: " ".join(x))
    nlp = spacy.load("en_core_web_lg")
    df_a['ABSTRACT'] = df_a['ABSTRACT'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))

    # сохранение результатов
    df_a.to_csv(df_final_path)

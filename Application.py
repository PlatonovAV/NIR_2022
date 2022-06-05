from pathlib import Path

import joblib
import nltk
import pandas as pd
import spacy
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer



nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
work_path = Path.cwd()
df_a_path = Path(work_path, 'dataset\\tested', 'test.csv')


# импорт данных
df_a = pd.read_csv(df_a_path, sep=',')

# Удаление ненужных столбцов
df_a.drop(['ID'], axis=1, inplace=True)
df_a.drop(['TITLE'], axis=1, inplace=True)

# удаление переносов
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

vectorizer1 = CountVectorizer(ngram_range=(1, 1))
X_df = vectorizer1.fit_transform(df_a["ABSTRACT"])
count_vect= pd.DataFrame(X_df.todense(), columns=vectorizer1.get_feature_names_out(), index=df_a.index)
df_a = pd.concat([df_a, count_vect], axis=1, join='inner')

clf =joblib.load('MultinomialNB.pkl')
df_tmp = pd.DataFrame()
for n in list(clf.feature_names_in_):
    if n not in df_tmp.columns:
        df_tmp = df_tmp.copy()
        df_tmp[n]= 0
for n in df_tmp.columns:
    if n in df_a.columns:
        df_tmp[n] = df_a[n]

df_a = df_tmp.copy()
df_a= df_a.fillna(0)
y_pred=clf.predict(df_a)





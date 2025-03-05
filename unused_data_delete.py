import pathlib
from pathlib import Path

import pandas as pd


# удаление текстов, принадлежащих не рассматриваемым категориям

def unused_data_delete():
    # загрузка данных
    work_path = pathlib.Path.cwd()
    df_a_path = Path(work_path, 'dataset\\clear\\step1', 'train.csv')
    df_final_path = Path(work_path, 'dataset\\clear\\step2', 'train.csv')
    df_a = pd.read_csv(df_a_path, sep=',', index_col=0)

    # удаление ненужных столбцов
    df_a.drop(["Quantitative Biology", "Quantitative Finance"], axis=1, inplace=True)

    # установка индекса
    df_a.reset_index(drop=True, inplace=True)
    df_a.index.names = ["ID"]

    # преобразование тематики текста к единому формату
    for index, row in df_a.iterrows():
        if row['Computer Science'] == 1 and row['Physics'] == 0 and row['Mathematics'] == 0 and row['Statistics'] == 0:
            df_a.loc[index, "LABEL"] = "Computer Science"
        elif row['Computer Science'] == 0 and row['Physics'] == 1 and row['Mathematics'] == 0 and row[
            'Statistics'] == 0:
            df_a.loc[index, "LABEL"] = "Physics"
        # elif row[0] == 0 and row[1]==0 and row[2]==1 and row[3]==0:
        #    df_a.loc[index, "LABEL"] = "Mathematics"
        # elif row[0] == 0 and row[1]==0 and row[2]==0 and row[3]==1:
        #    df_a.loc[index, "LABEL"] = "Statistics"
        else:
            df_a.drop(labels=[index], axis=0, inplace=True)
    df_a.reset_index(drop=True, inplace=True)

    # удаление оставшихся столбцов с тематикой текста
    df_a.drop(['Computer Science', 'Physics', 'Mathematics', 'Statistics'], axis=1, inplace=True)
    print(df_a['LABEL'].value_counts())

    # сохранение результатов в csv
    df_a.to_csv(df_final_path)

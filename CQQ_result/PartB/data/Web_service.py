import uvicorn as uvicorn
import fastapi
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import time
import sqlite3 as sql
import os
import requests
import pathlib


def build_SQL(root):
    """
    Preprocess and build the SQL dataset
    :param root:
    :return:
    """

    feature_df = pd.read_csv(os.path.join(root, 'data', 'features' + '.csv'))
    label_df = pd.read_csv(os.path.join(root, 'data', 'labels' + '.csv'))
    test_df = pd.read_csv(os.path.join(root, 'data', 'test' + '.csv'))

    df = pd.concat([feature_df, label_df['genre']], axis=1)
    df = df.reset_index(drop=True)

    df_train = df.dropna(axis=0, how='any')
    df_test = test_df.dropna(axis=0, how='any')
    print('There are total {} data removed from training dataset due to the missing value'.format(len(df) - len(df_train)))
    print('There are total {} data removed from training dataset due to the missing value'.format(
        len(test_df) - len(df_test)))

    removed_list = ['trackID', 'title', 'tags', 'duration']
    print('There are total {} columns removed due to the limited range value'.format(len(removed_list)))
    df_train = df_train.drop(removed_list, axis=1)
    df_test = df_test.drop(removed_list, axis=1)

    conn_train = sql.connect('train.db')
    df_train.to_sql('train', conn_train)

    conn_test = sql.connect('test.db')
    df_test.to_sql('test', conn_test)

    return conn_train, conn_test


class lightGBM_classifier:
    def __init__(self, model_file_path):

        labels = [
            "soul and reggae",
            "pop",
            "punk",
            "jazz and blues",
            "dance and electronica",
            "folk",
            "classic pop and rock",
            "metal"
        ]

        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(labels)}
        self.model_path = model_file_path

    def load_model(self):
        loaded_model = pickle.load(open(self.model_path, 'rb'))
        return loaded_model

    def predict(self, x):
        loaded_model = self.load_model()
        y_pred = loaded_model.predict_proba(x)
        y_pred = pd.DataFrame(np.argmax(y_pred, axis=1), columns=['genre'])
        y_pred = np.array(y_pred['genre'].apply(lambda x: self.id_to_label[x]))

        return y_pred


app = fastapi.FastAPI()


def SqltoDataFrame(conn_test):
    X_test = pd.read_sql(conn_test)
    return X_test


class Targets(str):
    classified_target1 = "soul and reggae"
    classified_target2 = "pop"
    classified_target3 = "punk"
    classified_target4 = "jazz and blues"
    classified_target5 = "dance and electronica"
    classified_target6 = "folk"
    classified_target7 = "classic pop and rock"
    classified_target8 = "metal"


class Item(BaseModel):
    input: str = None


@app.post('/classifier')
async def calculate(request_data: Item):
    root = request_data.input

    _, conn_test = build_SQL(root)
    df_test_x = SqltoDataFrame(conn_test)
    classifier = lightGBM_classifier(os.path.join(root, 'lightGBM_model.sav'))

    print("Detection for music classification! Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    result = classifier.predict(df_test_x)
    return result


def test(root):
    rec = requests.post("http://0.0.0.0:12455/classifier", data=root)
    return rec.text


if __name__ == '__main__':
    print('Loaded face model!')
    print("Service start!")
    root = pathlib.Path(os.path.abspath(__file__)).parent
    uvicorn.run(app=app, host="0.0.0.0", port=12455, workers=1)
    result = test(root)
    print(result)
import dill
import sys
from pathlib import Path
import pandas as pd
import json
import os

from pydantic import BaseModel
from datetime import datetime

path_fist = Path(sys.modules['__main__'].__file__).parents[1]
path_models = Path(path_fist) / 'data' / 'models'


def open_pkl(path_models):
   if len(os.listdir(path_models)) != 0:
       fpath = Path(path_models) / os.listdir(path_models)[0]
       return dill.load(open(str(fpath), 'rb'))
   else:
       exit()

model = open_pkl(path_models)

class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str

class Prediction(BaseModel):
    id: int
    pred: str
    price: int


def predict_form(form: Form):

    df = pd.DataFrame.from_dict([form.model_dump()])
    y = model.predict(df)

    return {
        'id': form.id,
        'pred': y[0],
        'price': form.price
    }


def load_json(json_file_path):

    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    return Form(**json_data)


def predict():

    # Указываем путь к папке с данными
    folder_data = Path(path_fist) / 'data' / 'test'
    folder_result = Path(path_fist) / 'data' / 'predictions'

    # Перебираем все файлы в папке
    results = []
    for file in folder_data.iterdir():

        # Загружаем файл json, делаем предсказание, сохраняем результат в переменную predicted_data
        predicted_data = predict_form(load_json(file))

        # Добавляем результаты в список results
        results.append(predicted_data)

        # Сохраняем список в DataFrame
        results_df = pd.DataFrame(results)

    # Сохраняем объединённый DataFrame в CSV-файл
    file_name = f'predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    output_filepath = folder_result / file_name
    results_df.to_csv(output_filepath, index=False)



if __name__ == '__main__':
    predict()

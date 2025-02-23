from urllib import response

import pandas as pd
import requests


# класс LoadData
class LoadData:
    def __init__(self):

        def load_csv(self, file_path):
            """
            Загружает CSV-файл по указанному пути с помощью библиотеки pandas.
            """
            # Загрузка файла с помощью pandas
            data = pd.read_csv(file_path)
            print(f"Файл {file_path} успешно загружен.")
            return data


        def load_json(self, file_path):
            """
            Загружает Json-файл по указанному пути с помощью библиотеки pandas.
            """
            # Загрузка файла с помощью pandas
            data = pd.read_json(file_path)
            print(f"Файл {file_path} успешно загружен.")
            return data


        def load_API(self, file_url):
            """
            Создает запрос GET.
            """
            data = requests.get(file_url)
            # Вывод кода
            print(response.status_code)
            # Вывод ответа, полученного от сервера API
            print(response.json())
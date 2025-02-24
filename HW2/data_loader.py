import pandas as pd
import requests

class LoadData:
    def __init__(self):
        pass  # Здесь не нужны вложенные функции

    def load_csv(self, file_path):
        """Загружает CSV-файл по указанному пути с помощью pandas."""
        df = pd.read_csv(file_path)
        print(f"Файл {file_path} успешно загружен.")
        return df

    def load_json(self, file_path):
        """Загружает JSON-файл по указанному пути с помощью pandas."""
        df = pd.read_json(file_path)
        print(f"Файл {file_path} успешно загружен.")
        return df

    def load_API(self, file_url):
        """Создает запрос GET к API."""
        response = requests.get(file_url)
        print(response.status_code)
        print(response.json())
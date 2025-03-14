import pandas as pd
import requests
import logging
import zipfile
import os

# Настраиваем логирование
logging.basicConfig(filename="app.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    encoding="utf-8")

class LoadData:
    def __init__(self):
        self.logger = logging.getLogger(__name__)  # Создаём логгер

    def load_csv(self, file_path):
        """Загружает CSV-файл по указанному пути с логированием ошибок."""
        try:
            df = pd.read_csv(file_path)
            print(f"Файл {file_path} успешно загружен.")
            self.logger.info(f"Файл {file_path} успешно загружен.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Ошибка: Файл {file_path} не найден.")
        except pd.errors.EmptyDataError:
            self.logger.error(f"Ошибка: Файл {file_path} пуст.")
        except pd.errors.ParserError:
            self.logger.error(f"Ошибка: Ошибка парсинга данных в {file_path}.")
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка при загрузке CSV: {e}")

    def load_json(self, file_path):
        """Загружает JSON-файл по указанному пути с логированием ошибок."""
        try:
            df = pd.read_json(file_path)
            self.logger.info(f"Файл {file_path} успешно загружен.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Ошибка: Файл {file_path} не найден.")
        except ValueError:
            self.logger.error(f"Ошибка: Неверный формат JSON в {file_path}.")
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка при загрузке JSON: {e}")

    def load_API(self, file_url):
        """Создает запрос GET к API с логированием ошибок."""
        try:
            response = requests.get(file_url)
            response.raise_for_status()  # Проверяем статус код ответа
            self.logger.info(f"Успешный запрос: {file_url}, статус {response.status_code}")
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP ошибка: {http_err}")
        except requests.exceptions.ConnectionError:
            self.logger.error("Ошибка соединения. Проверьте интернет.")
        except requests.exceptions.Timeout:
            self.logger.error("Ошибка: Превышено время ожидания.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка запроса: {e}")
    def load_from_zip(self, zip_path, file_name):
        """Извлекает и загружает CSV или JSON файл из ZIP-архива."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                if file_name in archive.namelist():
                    archive.extract(file_name, path="temp")
                    file_path = os.path.join("temp", file_name)

                    if file_name.endswith(".csv"):
                        return self.load_csv(file_path)
                    elif file_name.endswith(".json"):
                        return self.load_json(file_path)
                    else:
                        self.logger.error(f"Ошибка: Неподдерживаемый формат файла {file_name}.")
                else:
                    self.logger.error(f"Ошибка: Файл {file_name} не найден в архиве {zip_path}.")
        except zipfile.BadZipFile:
            self.logger.error(f"Ошибка: Файл {zip_path} не является корректным ZIP-архивом.")
        except FileNotFoundError:
            self.logger.error(f"Ошибка: Архив {zip_path} не найден.")
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка при работе с ZIP: {e}")
import pandas as pd
import requests
import logging

# Настраиваем логирование
logging.basicConfig(filename="app.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    encoding="utf-8")

class LoadData:
    def __init__(self):
        self.logger = logging.getLogger(__name__)  # Создаём логгер

    def load_csv(self, file_path):
        """Загружает CSV-файл по указанному пути с логированием ошибок."""
        try:
            df = pd.read_csv(file_path)
            print(f"Файл {file_path} успешно загружен.")
            self.logger.info(f"Файл {file_path} успешно загружен.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Ошибка: Файл {file_path} не найден.")
        except pd.errors.EmptyDataError:
            self.logger.error(f"Ошибка: Файл {file_path} пуст.")
        except pd.errors.ParserError:
            self.logger.error(f"Ошибка: Ошибка парсинга данных в {file_path}.")
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка при загрузке CSV: {e}")

    def load_json(self, file_path):
        """Загружает JSON-файл по указанному пути с логированием ошибок."""
        try:
            df = pd.read_json(file_path)
            self.logger.info(f"Файл {file_path} успешно загружен.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Ошибка: Файл {file_path} не найден.")
        except ValueError:
            self.logger.error(f"Ошибка: Неверный формат JSON в {file_path}.")
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка при загрузке JSON: {e}")

    def load_API(self, file_url):
        """Создает запрос GET к API с логированием ошибок."""
        try:
            response = requests.get(file_url)
            response.raise_for_status()  # Проверяем статус код ответа
            self.logger.info(f"Успешный запрос: {file_url}, статус {response.status_code}")
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP ошибка: {http_err}")
        except requests.exceptions.ConnectionError:
            self.logger.error("Ошибка соединения. Проверьте интернет.")
        except requests.exceptions.Timeout:
            self.logger.error("Ошибка: Превышено время ожидания.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка запроса: {e}")

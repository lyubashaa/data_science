from HW2 import data_loader as dl

# указание пути к CSV-файлу
file_path = '/Users/lubovsoldatenko/DataspellProjects/datascience/HW2/data/water_potability.csv'

# загрузка данных
loader = dl.LoadData()
df = loader.load_csv(file_path)

# просмотр данных
df.head()

# информация о данных
df.info()

# базовая статистика
df.describe()




from HW2 import data_loader as dl
from HW2.data_analyzer import MissingValuesHandler
from HW2.data_visualizer import DataVisualizer

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

# Создаем объект обработчика
handler = MissingValuesHandler(df)

# Подсчет пропущенных значений
handler.count_missing_values()

# Отчет о пропущенных данных
handler.missing_values_report()

# Заполняем пропущенные значения
handler.fill_missing_values()

# Удаляем строки с пропущенными значениями (если они остались)
handler.drop_missing_values()

# Проверяем результат
print(df)

# Создаем объект визуализатора
visualizer = DataVisualizer(df)

#  Построение гистограммы
visualizer.plot_histogram(column='Sulfate', bins=20, color='blue', save=True)

# Строим линейный график с ограничением на 50 точек
visualizer.plot_line_chart(x_column='ph', y_column='Hardness', color='green', save=True, limit=50, markers=False)

#  Построение диаграммы рассеяния
visualizer.plot_scatter(x_column='ph', y_column='Hardness', color='red', save=True)


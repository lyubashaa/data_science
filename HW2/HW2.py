import data_loader as dl

# указание пути к CSV-файлу
file_path = '/Users/lubovsoldatenko/venv/ds/data_science/HW2/Choclate Quality analysis.csv'

# загрузка данных
data = dl.load_csv(file_path)

# просмотр данных
data.head()

# информация о данных
data.info()

# базовая статистика
data.describe()


import data_visualizer as dv

# Создание экземпляра визуализатора
visualizer = dv.DataVisualizer(data)

# Добавление гистограммы
visualizer.add_histogram('sepal length (cm)', bins=20, color='blue', label='Sepal Length')
visualizer.show()

# Добавление линейного графика
visualizer.add_line_plot('sepal length (cm)', 'petal length (cm)', color='green', label='Sepal vs Petal Length')
visualizer.show()

# Добавление диаграммы рассеяния
visualizer.add_scatter_plot('sepal width (cm)', 'petal width (cm)', color='red', label='Sepal vs Petal Width')
visualizer.show()







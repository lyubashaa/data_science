import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    def __init__(self, df, save_dir="plots"):
        """
        Класс для визуализации данных.

        :param df: DataFrame с данными
        :param save_dir: Директория для сохранения графиков
        """
        self.df = df
        # Определяем путь к директории, где находится сам скрипт
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Если директория для сохранения не задана, создаем "plots" рядом с файлом
        self.save_dir = save_dir if save_dir else os.path.join(script_dir, "plots")

        # Создаем папку, если ее нет
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_histogram(self, column, bins=10, color='blue', save=False):
        """
        Строит гистограмму для указанного столбца.

        :param column: Название столбца
        :param bins: Количество бинов (столбцов) в гистограмме
        :param color: Цвет гистограммы
        :param save: Флаг для сохранения графика
        """
        if column not in self.df.columns:
            print(f"Ошибка: '{column}' отсутствует в DataFrame.")
            return

        plt.figure(figsize=(10, 5))
        sns.histplot(self.df[column].dropna(), bins=bins, color=color, kde=True)
        plt.xlabel(column)
        plt.ylabel("Частота")
        plt.title(f'Гистограмма: {column}')
        plt.grid(True)

        if save:
            plot_path = os.path.join(self.save_dir, f'histogram_{column}.png')
            plt.savefig(plot_path)
            print(f"График сохранен в {plot_path}")

        plt.show()

    def plot_scatter(self, x_column, y_column, color='red', save=False):
        """
        Строит диаграмму рассеяния.

        :param x_column: Название столбца для оси X
        :param y_column: Название столбца для оси Y
        :param color: Цвет точек
        :param save: Флаг для сохранения графика
        """
        if x_column not in self.df.columns or y_column not in self.df.columns:
            print(f"Ошибка: '{x_column}' или '{y_column}' отсутствует в DataFrame.")
            return

        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=self.df, x=x_column, y=y_column, color=color, alpha=0.7)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'Диаграмма рассеяния: {x_column} vs {y_column}')
        plt.grid(True)

        if save:
            plot_path = os.path.join(self.save_dir, f'scatter_{x_column}_vs_{y_column}.png')
            plt.savefig(plot_path)
            print(f"График сохранен в {plot_path}")

        plt.show()

    def plot_line_chart(self, x_column, y_column, color='blue', save=False, limit=None, markers=False):
        """
        Строит линейный график.

        :param x_column: Название столбца для оси X
        :param y_column: Название столбца для оси Y
        :param color: Цвет линии
        :param save: Флаг для сохранения графика
        :param limit: Количество первых значений для отображения (если None - все)
        :param markers: Флаг для отображения маркеров на точках
        """
        if x_column not in self.df.columns or y_column not in self.df.columns:
            print(f"Ошибка: '{x_column}' или '{y_column}' отсутствует в DataFrame.")
            return

        data = self.df[[x_column, y_column]].dropna()
        data = data.sort_values(by=x_column)  # Сортируем по X

        if limit is not None:
            data = data.head(limit)  # Берем только первые `limit` значений

        plt.figure(figsize=(10, 5))
        plt.plot(data[x_column], data[y_column], color=color, marker='o' if markers else '', linestyle='-')

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'Линейный график: {x_column} vs {y_column}')
        plt.grid(True)

        if save:
            plot_path = os.path.join(self.save_dir, f'line_chart_{x_column}_vs_{y_column}.png')
            plt.savefig(plot_path)
            print(f"График сохранен в {plot_path}")

        plt.show()

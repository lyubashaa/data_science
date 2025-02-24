import matplotlib.pyplot as plt
import seaborn as sns

# установка стиля Seaborn для красивых графиков
sns.set(style="whitegrid")


class DataVisualizer:

    def __init__(self, df):
        self.df = df

    def hist(df):
        # создание гистограмм для каждой числовой переменной
        df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')

        # Добавление названий для каждого графика и осей
        for ax in plt.gcf().get_axes():
            ax.set_xlabel('Значение')
            ax.set_ylabel('Частота')
            ax.set_title(ax.get_title().replace('wine_class', 'Класс вина'))

        # Регулировка макета для предотвращения наложения подписей
        plt.tight_layout()

        # Показать график
        plt.show()
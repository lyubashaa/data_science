import pandas as pd

class MissingValuesHandler:
    def __init__(self, df):
        """
        Класс для работы с пропущенными значениями в DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Ошибка: Переданный объект не является DataFrame.")

        self.df = df

    def count_missing_values(self):
        """
        Подсчитывает количество пропущенных значений в каждом столбце.
        """
        return self.df.isnull().sum()

    def missing_values_report(self):
        """
        Выводит отчет о пропущенных значениях.
        """
        missing_counts = self.count_missing_values()
        total_values = self.df.shape[0]
        missing_percentage = (missing_counts / total_values) * 100

        report = pd.DataFrame({
            'Missing Values': missing_counts,
            'Missing Percentage': missing_percentage
        })

        print("\n Отчет о пропущенных значениях:")
        print(report)

    def fill_missing_values(self):
        """
        Заполняет пропущенные значения:
        - Числовые столбцы → медианное значение
        - Категориальные (объектные) столбцы → самое частое значение (мода)
        """
        num_cols = self.df.select_dtypes(include=['number']).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median(numeric_only=True))

        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        print("\n Пропущенные значения заполнены (числовые — медианой, строковые — модой).")

    def drop_missing_values(self):
        """
        Удаляет все строки с пропущенными значениями.
        """
        before_drop = self.df.shape[0]
        self.df.dropna(inplace=True)
        after_drop = self.df.shape[0]

        print(f"\n Удалено {before_drop - after_drop} строк с пропущенными значениями.")

    def remove_columns(self, columns_to_remove):
        """Удаляет указанные столбцы из DataFrame."""
        self.df = self.df.drop(columns=columns_to_remove, errors='ignore')
        return self.df  # Возвращаем обновленный DataFrame

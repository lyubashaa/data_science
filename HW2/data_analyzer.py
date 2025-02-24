import pandas as pd

def count_missing_values(df):
    """
    Подсчитывает количество пропущенных значений в каждом столбце DataFrame.
    """
    return df.isnull().sum()

def missing_values_report(df):
    """
    Выводит отчет о пропущенных значениях в DataFrame.
    """
    missing_counts = count_missing_values(df)
    total_values = df.shape[0]
    missing_percentage = (missing_counts / total_values) * 100

    report = pd.DataFrame({
        'Missing Values': missing_counts,
        'Missing Percentage': missing_percentage
    })

    print(report)

def fill_missing_values(df):
    """
    Заполняет пропущенные значения в DataFrame медианным значением.
    """
df.fillna(df.median(), inplace=True)
from HW3.data_loader import LoadData
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from HW3.data_visualizer import DataVisualizer

# указание пути к CSV-файлу
file_path = '/HW3/data/Mall_Customers.csv'

# загрузка данных
loader = LoadData()
df = loader.load_csv(file_path)

# Подключаемся к базе данных (или создаем, если нет)
conn = sqlite3.connect("my_database.db")
cursor = conn.cursor()

# Проверяем, что это DataFrame
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

# Записываем данные в таблицу (если таблицы нет, создастся автоматически)
df.to_sql("customers", conn, if_exists="append", index=False)

print("Данные успешно добавлены.")

# Выбор клиентов, у которых ежегодный доход превышает 30k$.
cursor.execute('SELECT CustomerID, Genre, "Annual Income (k$)" FROM customers WHERE "Annual Income (k$)" > 30')
result = cursor.fetchall()
df = result
for row in result:
    print(row)

# Получаем названия столбцов
columns = [desc[0] for desc in cursor.description]

# Преобразуем в DataFrame
df = pd.DataFrame(result, columns=columns)

print(df.head())  # Проверяем результат

# Настраиваем стиль
sns.set_theme(style="white grid")

# Создаем график
plt.figure(figsize=(8, 5))
sns.barplot(x=df["Genre"], y=df["Annual Income (k$)"], palette="cool warm")

# Добавляем заголовок
plt.title("Годовой доход клиентов по полу", fontsize=14)
plt.xlabel("Пол")
plt.ylabel("Annual Income (k$)")

# Показываем график
plt.show()

# Выбор клиентов по возрастанию дохода, у которых средняя оценка трат превышает значение 50.
cursor.execute('SELECT CustomerID, "Annual Income (k$)", "Spending Score (1-100)" FROM customers WHERE "Spending Score (1-100)" > 50 ORDER BY "Annual Income (k$)" ASC')
result = cursor.fetchall()
for row in result:
    print(row)

# Получаем названия столбцов
columns = [desc[0] for desc in cursor.description]

# Преобразуем в DataFrame
df = pd.DataFrame(result, columns=columns)

print(df.head())  # Проверяем результат

# Создаем объект визуализатора
visualizer = DataVisualizer(df)

# Строим линейный график с ограничением на 100 точек
visualizer.plot_line_chart(x_column='Annual Income (k$)', y_column='Spending Score (1-100)', color='blue', save=True, limit=100, markers=False)

# Общий доход всех клиентов.
cursor.execute('SELECT SUM("Annual Income (k$)") AS Total_Income FROM customers')
result = cursor.fetchall()
for row in result:
    print(row)

# Общий доход всех клиентов с гендерным разделением.
cursor.execute('SELECT SUM("Annual Income (k$)") AS Total_Income FROM customers GROUP BY Genre')
result = cursor.fetchall()
for row in result:
    print(row)

# Средний доход всех клиентов.
cursor.execute('SELECT AVG("Annual Income (k$)") AS Average_Income FROM customers')
result = cursor.fetchall()
for row in result:
    print(row)

# Средний доход клиентов с высоким уровнем трат (>50).
cursor.execute('SELECT AVG("Annual Income (k$)") AS Average_Income FROM customers WHERE "Spending Score (1-100)" > 50')
result = cursor.fetchall()
for row in result:
    print(row)

query = 'SELECT "Annual Income (k$)", "Spending Score (1-100)" FROM customers'
df = pd.read_sql(query, conn)

plt.figure(figsize=(10, 5))
sns.lineplot(x=df["Annual Income (k$)"], y=df["Spending Score (1-100)"], marker="o")

plt.title("График зависимости трат от дохода клиентов")
plt.ylabel("Оценка трат")
plt.xlabel("Доход (k$)")
plt.show()

conn.commit()
conn.close()
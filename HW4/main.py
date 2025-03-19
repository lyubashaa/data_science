import data_loader as dl
from HW4.data_analyzer import MissingValuesHandler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Указание пути к CSV-файлу
file_path = '/Users/lubovsoldatenko/DataspellProjects/datascience/HW4/data/Traffic.csv'

# Загрузка данных
loader = dl.LoadData()
df = loader.load_csv(file_path)

# Просмотр данных
df.head(5)

# Информация о данных
df.info()

# Базовая статистика
df.describe()

# Количество значений в датасете
len(df)

# Создаем объект обработчика
handler = MissingValuesHandler(df)

# Подсчет пропущенных значений
handler.count_missing_values()

# Отчет о пропущенных данных
handler.missing_values_report()

# Поиск дубликатов
handler.find_duplicates()

# Проверка, количесва данных в каждом классе, чтобы понять, насколько сбалансирован датасет
df['Traffic Situation'].value_counts()

# Балансировка данных
# Определяем целевой столбец
target_column = 'Traffic Situation'

# Балансируем, оставляя ровно 304 элемента в каждом классе
df_balanced = df.groupby(target_column).sample(n=304, random_state=42)

# Проверяем результат
print(df_balanced[target_column].value_counts())

# Преобразование времени в datetime
df_balanced['Time'] = pd.to_datetime(df_balanced['Time'])

# Группировка данных по часам
df_balanced['Hour'] = df_balanced['Time'].dt.hour

# Суммирование по каждому виду транспорта для каждого часа
hourly_traffic = df_balanced.groupby('Hour')[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].sum().reset_index()


# Построение линейных графиков для каждого вида транспорта в зависимости от времени
plt.figure(figsize=(10, 6))

for column in ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']:
    plt.plot(hourly_traffic['Hour'], hourly_traffic[column], label=column)

# Оформление графика
plt.title('Зависимость видов транспорта от времени')
plt.xlabel('Часы')
plt.ylabel('Количество транспорта')
plt.legend()
plt.grid()
plt.show()

# Кодирование категориальных признаков
df_balanced['Time'] = df_balanced['Time'].astype('category').cat.codes
df_balanced['Traffic Situation'] = df_balanced['Traffic Situation'].astype('category').cat.codes
df_balanced['Day of the week'] = df_balanced['Day of the week'].astype('category').cat.codes
df_balanced['Hour'] = df_balanced['Hour'].astype('category').cat.codes

# Матрица корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(df_balanced.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица признаков')
plt.show()

# Ящик с усами
plt.figure(figsize=(10, 8))
sns.boxplot(data=df_balanced)
plt.xticks(rotation=90)
plt.title('График для поиска выбросов')
plt.show()

# Поиск выбросов для параметра BikeCount
Q1 = df_balanced['BikeCount'].quantile(0.25)
Q3 = df_balanced['BikeCount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_balanced[(df_balanced['BikeCount'] < lower_bound) | (df_balanced['BikeCount'] > upper_bound)]
print(f'Найдено {len(outliers)} выбросов')


# Удаляем выбросы из набора данных
df_balanced = df_balanced[(df_balanced['BikeCount'] >= lower_bound) & (df_balanced['BikeCount'] <= upper_bound)]
print(f'Очищенный набор данных содержит {len(df_balanced)} записей')

# Стандартизация
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(df_balanced)
print("Standard scaled:\n", X_standard)

# Разделяем на признаки (X) и целевую переменную (y)
X = df_balanced.drop(columns=["Traffic Situation"])
y = df_balanced["Traffic Situation"]

# Разделяем на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ANOVA F-тест
k = 3  # Количество признаков для отбора
selector = SelectKBest(score_func=f_classif, k=k)
X_train_best = selector.fit_transform(X_train, y_train)
X_test_best = selector.transform(X_test)

# Получение имен выбранных признаков
selected_features = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_features]
print(f"\nОтобранные признаки с помощью SelectKBest (k={k}):")
print(selected_feature_names.tolist())

# Разделяем на признаки (X) и целевую переменную (y)
X = df_balanced.drop(columns=["Traffic Situation"])
y = df_balanced["Traffic Situation"]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)

# Обучение модели на обучающем наборе данных
gb_classifier.fit(X_train, y_train)

# Предсказание классов на тестовом наборе данных
y_pred = gb_classifier.predict(X_test)

# Оценка производительности модели
accuracy_gb = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_gb}")

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Поиск лучших параметров
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Разделяем на признаки (X) и целевую переменную (y)
X = df_balanced.drop(columns=["Traffic Situation"])
y = df_balanced["Traffic Situation"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание и обучение классификатора Extra Trees
clf = ExtraTreesClassifier(n_estimators=100, max_features='sqrt', random_state=42)
clf.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = clf.predict(X_test)

# Оценка производительности модели
accuracy_extra_trees = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_extra_trees}")

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделяем на признаки (X) и целевую переменную (y)
X = df_balanced.drop(columns=["Traffic Situation"])
y = df_balanced["Traffic Situation"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели QDA
qda_classifier = QuadraticDiscriminantAnalysis()

# Обучение модели на обучающем наборе данных
qda_classifier.fit(X_train, y_train)

# Предсказание классов на тестовом наборе данных
y_pred = qda_classifier.predict(X_test)

# Оценка производительности модели
accuracy_qda = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_qda}")

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделяем на признаки (X) и целевую переменную (y)
X = df_balanced.drop(columns=["Traffic Situation"])
y = df_balanced["Traffic Situation"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Настройка гиперпараметра k с использованием GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)

# Вывод лучшего значения k
print("Лучшее значение k:", knn_cv.best_params_['n_neighbors'])

# Обучение модели с лучшим значением k
knn_best = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
knn_best.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = knn_best.predict(X_test)

# Оценка производительности модели
accuracy_knn_best = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_knn_best}")

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Разделяем на признаки (X) и целевую переменную (y)
X = df_balanced.drop(columns=["Traffic Situation"])
y = df_balanced["Traffic Situation"]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели CatBoostClassifier
clf = CatBoostClassifier(iterations=50, depth=10, learning_rate=0.2, loss_function='MultiClass', random_state=42)

# Обучение модели на обучающем наборе данных
clf.fit(X_train, y_train)

# Предсказание классов на тестовом наборе данных
y_pred = clf.predict(X_test)

# Оценка производительности модели
accuracy_catboost = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_catboost}")

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Создание новых признаков
# Статистические признаки (среднее, медиана, стандартное отклонение)
X["mean"] = X.mean(axis=1)
X["std"] = X.std(axis=1)
X["median"] = X.median(axis=1)

# Объединяем новые признаки
X_new = np.hstack(X[["mean", "std", "median"]].values)

# Разделяем на признаки (X) и целевую переменную (y)
X_new = df_balanced.drop(columns=["Traffic Situation"])
y = df_balanced["Traffic Situation"]

# Разделение данных на обучающий и тестовый наборы
X_new_train, X_new_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Стандартизируем данные (масштабируем признаки)
scaler = StandardScaler()
X_new_train_scaled = scaler.fit_transform(X_new_train)
X_new_test_scaled = scaler.transform(X_new_test)

# Создадим и обучим модель логистической регрессии
model = LogisticRegression(
    random_state=42,
    solver='lbfgs',
    max_iter=200
)
model.fit(X_new_train_scaled, y_train)

# Сделаем предсказания на тестовом наборе данных
y_pred = model.predict(X_new_test_scaled)

# Оценка производительности модели
accuracy_LogisticRegression = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_LogisticRegression}")

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Создаем список классификаторов
classifiers_names = ["Gradient Boosting", "Extra Trees", "QDA", "KNeighbors", "CatBoost", "LogisticRegression"]
accuracies = [accuracy_gb, accuracy_extra_trees, accuracy_qda, accuracy_knn_best, accuracy_catboost, accuracy_LogisticRegression]

# Визуализация результатов
plt.figure(figsize=(8, 5))
plt.bar(classifiers_names, accuracies)
plt.xlabel("Классификаторы")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Ограничиваем диапазон
plt.title("Сравнение точности классификаторов")
plt.show()


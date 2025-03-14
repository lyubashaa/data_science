import data_loader as dl
from HW4.data_analyzer import MissingValuesHandler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
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
from sklearn.dummy import DummyClassifier

# Указание пути к CSV-файлу
file_path = '/Users/lubovsoldatenko/DataspellProjects/datascience/HW4/data/Traffic.csv'

# Загрузка данных
loader = dl.LoadData()
df = loader.load_csv(file_path)

# Просмотр данных
df.head(5)

# Создаем объект обработчика
handler = MissingValuesHandler(df)

# Удаляем столбцы
df = handler.remove_columns(['Date', 'Day of the week'])
print("\nDataFrame после удаления столбцов")
print(df)

# Информация о данных
df.info()

# Базовая статистика
df.describe()

# Количество значений в датасете
len(df)

# Подсчет пропущенных значений
handler.count_missing_values()

# Отчет о пропущенных данных
handler.missing_values_report()

#Визуализация для проверки распределения классов

# Предположим, что у нас есть два числовых признака и колонка 'Traffic Situation' с 4 категориями
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='CarCount', y='Total', hue='Traffic Situation', style='Traffic Situation', palette='Set1')
plt.title("Классификация загруженности дорог ")
plt.xlabel("CarCount")
plt.ylabel("Total")
plt.legend(title="Загруженность дорог")
plt.show()

# Проверка, количесва данных в каждом классе, чтобы понять, насколько сбалансирован датасет
df['Traffic Situation'].value_counts()

# Величина загруженности дорог
plt.figure(figsize=(6, 4))
sns.countplot(x='Traffic Situation', data=df)
plt.title('Величина загруженности дорог')
plt.xlabel('Вид загруженности')
plt.ylabel('Величина загруженности дорог')
plt.show()

# Кодирование категориальных признаков
df['Time'] = df['Time'].astype('category').cat.codes
df['Traffic Situation'] = df['Traffic Situation'].astype('category').cat.codes

# Матрица корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица признаков')
plt.show()

# Ящик с усами
plt.figure(figsize=(10, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('График для поиска выбросов')
plt.show()

# Поиск выбросов для параметра BikeCount
Q1 = df['BikeCount'].quantile(0.25)
Q3 = df['BikeCount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['BikeCount'] < lower_bound) | (df['BikeCount'] > upper_bound)]
print(f'Найдено {len(outliers)} выбросов')

# Применяем RobustScaler
scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("До масштабирования:\n", df)
print("\nПосле RobustScaler:\n", df_scaled)

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Traffic Situation"])
y = df["Traffic Situation"]

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
X = df.drop(columns=["Traffic Situation"])
y = df["Traffic Situation"]

# Создание новых признаков
# Статистические признаки (среднее, медиана, стандартное отклонение)
X["mean"] = X.mean(axis=1)
X["std"] = X.std(axis=1)
X["median"] = X.median(axis=1)

# Объединяем новые признаки
X_new = np.hstack(X[["mean", "std", "median"]].values)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.01,
                                           subsample=0.8, max_features="sqrt", random_state=42)

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

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)
best_model = grid_search.best_estimator_

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

# QDA
# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Traffic Situation"])
y = df["Traffic Situation"]

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

# KNeighborsClassifier
# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Traffic Situation"])
y = df["Traffic Situation"]

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

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

# Dummy Classifier с параметром most_frequent
# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Traffic Situation"])
y = df["Traffic Situation"]

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Создание и обучение модели Decision Tree Classifier

# Создание и обучение Dummy Classifier
dummy_clf = DummyClassifier(strategy="stratified") # stratified
dummy_clf.fit(X_train, y_train)

# Оценка модели
y_pred = dummy_clf.predict(X_test)

# Оценка производительности модели
accuracy_dummy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_dummy}")

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Создаем список классификаторов
classifiers_names = ["Gradient Boosting", "Extra Trees", "QDA", "KNeighbors", "Dummy Classifier"]
accuracies = [accuracy_gb, accuracy_extra_trees, accuracy_qda, accuracy_knn_best, accuracy_dummy]

# Визуализация результатов
plt.figure(figsize=(8, 5))
plt.bar(classifiers_names, accuracies)
plt.xlabel("Классификаторы")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Ограничиваем диапазон
plt.title("Сравнение точности классификаторов")
plt.show()
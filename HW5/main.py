import kagglehub
import data_loader as dl
from HW5.data_analyzer import MissingValuesHandler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

# Download latest version
path = kagglehub.dataset_download("equilibriumm/sleep-efficiency")

print("Path to dataset files:", path)

# Указание пути к CSV-файлу
file_path = '/Users/lubovsoldatenko/DataspellProjects/datascience/HW5/data/Sleep_Efficiency.csv'

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

# Заполняем пропущенные значения
handler.fill_missing_values()

# Удаляем строки с пропущенными значениями (если они остались)
handler.drop_missing_values()

# Поиск дубликатов
handler.find_duplicates()

df['Bedtime'] = pd.to_datetime(df['Bedtime'])
df['Wakeup time'] = pd.to_datetime(df['Wakeup time'])

# Кодирование категориальных признаков
df['Smoking status'] = df['Smoking status'].map({'No': 0, 'Yes': 1})
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Матрица корреляций
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица признаков')
plt.show()

#Удаление признаков с высокой корреляцией
df = df.drop(columns=['Wakeup time'])
df = df.drop(columns=['Light sleep percentage'])

sns.histplot(df['Sleep efficiency'], kde=True)
plt.title("Distribution of target variable 'Sleep efficiency'")
plt.show()

# Ящик с усами
plt.figure(figsize=(10, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('График для поиска выбросов')
plt.show()

# Поиск выбросов для параметра Sleep duration
Q1 = df['Sleep duration'].quantile(0.25)
Q3 = df['Sleep duration'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Sleep duration'] < lower_bound) | (df['Sleep duration'] > upper_bound)]
print(f'Найдено {len(outliers)} выбросов')

# Поиск выбросов для параметра Deep sleep percentage
Q1 = df['Deep sleep percentage'].quantile(0.25)
Q3 = df['Deep sleep percentage'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Deep sleep percentage'] < lower_bound) | (df['Deep sleep percentage'] > upper_bound)]
print(f'Найдено {len(outliers)} выбросов')

# Поиск выбросов для параметра Caffeine consumption
Q1 = df['Caffeine consumption'].quantile(0.25)
Q3 = df['Caffeine consumption'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Caffeine consumption'] < lower_bound) | (df['Caffeine consumption'] > upper_bound)]
print(f'Найдено {len(outliers)} выбросов')

# Удаление выбросов
df = df[(df['Sleep duration'] >= lower_bound) & (df['Sleep duration'] <= upper_bound)]
df = df[(df['Deep sleep percentage'] >= lower_bound) & (df['Deep sleep percentage'] <= upper_bound)]
df = df[(df['Caffeine consumption'] >= lower_bound) & (df['Caffeine consumption'] <= upper_bound)]
print(f'Очищенный набор данных содержит {len(df)} записей')

df['Bedtime'] = df['Bedtime'].apply(lambda x: x.hour + (0.5 if x.time().minute > 0 else 0))

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Подбираем оптимальный уровень регуляризации
alphas = np.logspace(-3, 3, 10)
best_alpha = None
best_score = -np.inf

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

# Обучаем модель с оптимальным alpha
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train, y_train)

# Оцениваем модель
train_score = ridge.score(X_train, y_train)
test_score = ridge.score(X_test, y_test)

# Выводим важность признаков
feature_importance = np.abs(ridge.coef_)
best_features = np.argsort(feature_importance)[::-1]

print(f"Optimal alpha: {best_alpha:.4f}")
print(f"Train R^2 Score: {train_score:.4f}")
print(f"Test R^2 Score: {test_score:.4f}")
print("Top features by importance:")
for i in best_features:
    print(f"Feature {i}: {feature_importance[i]:.4f}")

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели Gradient Boosting Regressor
model = GradientBoostingRegressor() # параметры
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred_gbr = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_gbr)
mse = mean_squared_error(y_test, y_pred_gbr)
r2 = r2_score(y_test, y_pred_gbr)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Создание и обучение модели Extra Trees Regressor
etr = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
etr.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred_etr = etr.predict(X_test)

# 9. Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_etr)
mse = mean_squared_error(y_test, y_pred_etr)
r2 = r2_score(y_test, y_pred_etr)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Создание и обучение модели Random Forest Regressor
rfr = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rfr.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred_rfr = rfr.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_rfr)
mse = mean_squared_error(y_test, y_pred_rfr)
r2 = r2_score(y_test, y_pred_rfr)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели CatBoost для регрессии
regressor = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=6, random_state=42, verbose=0)
regressor.fit(X_train, y_train)

# Выполнение предсказаний на тестовых данных
y_pred_cbr = regressor.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_cbr)
mse = mean_squared_error(y_test, y_pred_cbr)
r2 = r2_score(y_test, y_pred_cbr)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели AdaBoostRegressor
regressor = AdaBoostRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Выполнение предсказаний на тестовых данных
y_pred_abr = regressor.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_abr)
mse = mean_squared_error(y_test, y_pred_abr)
r2 = r2_score(y_test, y_pred_abr)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели регрессии Lasso
regressor = Lasso(alpha=0.1)  # Здесь alpha - гиперпараметр регуляризации L1
regressor.fit(X_train, y_train)

# Выполнение предсказаний на тестовых данных
y_pred_l = regressor.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_l)
mse = mean_squared_error(y_test, y_pred_l)
r2 = r2_score(y_test, y_pred_l)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Создание и обучение модели Ridge Regression
ridge_reg = Ridge(alpha=1.0, random_state=42)
ridge_reg.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred_r = ridge_reg.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_r)
mse = mean_squared_error(y_test, y_pred_r)
r2 = r2_score(y_test, y_pred_r)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель K-ближайших соседей
knn_regressor = KNeighborsRegressor(n_neighbors=5) # для предсказания значения целевой переменной для нового наблюдения будет использоваться 5 ближайших соседей из обучающей выборки.

# Обучаем модель K-ближайших соседей
knn_regressor.fit(X_train, y_train)

# Делаем прогнозы на тестовом наборе
y_pred_knn = knn_regressor.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_knn)
mse = mean_squared_error(y_test, y_pred_knn)
r2 = r2_score(y_test, y_pred_knn)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop(columns=["Sleep efficiency"])
y = df["Sleep efficiency"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Обучаем модель ElasticNet
elastic_net.fit(X_train, y_train)

# Делаем прогнозы на тестовом наборе
y_pred_en = elastic_net.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred_en)
mse = mean_squared_error(y_test, y_pred_en)
r2 = r2_score(y_test, y_pred_en)

print(f'Метрики качества:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Словарь с моделями и их предсказаниями
models_predictions = {
    "GradientBoostingRegressor": y_pred_gbr,
    "ExtraTreesRegressor": y_pred_etr,
    "RandomForestRegressor": y_pred_rfr,
    "CatBoostRegressor": y_pred_cbr,
    "AdaBoostRegressor": y_pred_abr,
    "Lasso": y_pred_l,
    "Ridge": y_pred_r,
    "KNeighborsRegressor": y_pred_knn,
    "ElasticNet": y_pred_en,
}

# Список для хранения метрик
results = []

# Вычисление метрик для каждой модели
for model_name, y_pred in models_predictions.items():
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append([model_name, mae, mse, rmse, r2])

# Создание DataFrame
metrics_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R² Score"])

# Сохранение в CSV-файл
metrics_df.to_csv("regression_models_comparison.csv", index=False)

print("Файл 'regression_models_comparison.csv' успешно создан!")

# Построение графика Фактические vs предсказанные значения
plt.figure(figsize=(15, 15))
plt.subplot(3, 3, 1)
sns.scatterplot(x=y_test, y=y_pred_gbr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('GradientBoostingRegressor')

plt.subplot(3, 3, 2)
sns.scatterplot(x=y_test, y=y_pred_etr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('ExtraTreesRegressor')

plt.subplot(3, 3, 3)
sns.scatterplot(x=y_test, y=y_pred_rfr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('RandomForestRegressor')

plt.subplot(3, 3, 4)
sns.scatterplot(x=y_test, y=y_pred_cbr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('CatBoostRegressor')

plt.subplot(3, 3, 5)
sns.scatterplot(x=y_test, y=y_pred_abr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('AdaBoostRegressor')

plt.subplot(3, 3, 6)
sns.scatterplot(x=y_test, y=y_pred_l)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('Lasso')

plt.subplot(3, 3, 7)
sns.scatterplot(x=y_test, y=y_pred_r)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('Ridge')

plt.subplot(3, 3, 8)
sns.scatterplot(x=y_test, y=y_pred_knn)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title('KNeighborsRegressor')

plt.subplot(3, 3, 9)
sns.scatterplot(x=y_test, y=y_pred_en)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True values')
plt.ylabel('Фактические vs предсказанные значения')
plt.title(''
          'ElasticNet')

plt.tight_layout()
plt.show()

# Построение графика Остатки vs предсказанные значения
plt.figure(figsize=(15, 15))
plt.subplot(3, 3, 1)
sns.scatterplot(x=y_test, y=y_pred_gbr-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('GradientBoostingRegressor')

plt.subplot(3, 3, 2)
sns.scatterplot(x=y_test, y=y_pred_etr-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('ExtraTreesRegressor')

plt.subplot(3, 3, 3)
sns.scatterplot(x=y_test, y=y_pred_rfr-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('RandomForestRegressor')

plt.subplot(3, 3, 4)
sns.scatterplot(x=y_test, y=y_pred_cbr-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('CatBoostRegressor')

plt.subplot(3, 3, 5)
sns.scatterplot(x=y_test, y=y_pred_abr-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('AdaBoostRegressor')

plt.subplot(3, 3, 6)
sns.scatterplot(x=y_test, y=y_pred_l-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('Lasso')

plt.subplot(3, 3, 7)
sns.scatterplot(x=y_test, y=y_pred_r-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('Ridge')

plt.subplot(3, 3, 8)
sns.scatterplot(x=y_test, y=y_pred_knn-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('KNeighborsRegressor')

plt.subplot(3, 3, 9)
sns.scatterplot(x=y_test, y=y_pred_en-y_test)
sns.lineplot(x=[0.5, 1], y=[0, 0], color='green')
plt.xlabel('True values')
plt.ylabel('Остатки vs предсказанные значения')
plt.title('ElasticNet')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_gbr, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('GradientBoostingRegressor')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_etr, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('ExtraTreesRegressor')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_rfr, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('RandomForestRegressor')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_cbr, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('CatBoostRegressor')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_abr, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('AdaBoostRegressor')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_l, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('Lasso')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_r, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('Ridge')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_knn, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('KNeighborsRegressor')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=X_test.index, y=y_test, color='red', alpha=0.2, label='True values')
sns.scatterplot(x=X_test.index, y=y_pred_en, color='black', alpha=0.5, label='Predicted values')
plt.legend(loc='upper right')
plt.title('ElasticNet')
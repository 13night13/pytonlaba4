import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Загрузка предварительно обработанных данных Titanic из CSV-файла
train_data = pd.read_csv(r'D:\labi\lab1\.venv\processed_titanic.csv')

# Разделение данных на признаки (X) и целевую переменную (Y)
X = train_data.drop(['PassengerId', 'Transported'], axis=1)  # Удаляем ненужные столбцы
Y = train_data['Transported']  # Целевая переменная — был ли человек транспортирован

# Делим данные на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Функция для обучения модели и оценки её качества
def train_and_evaluate(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)  # Обучаем модель на тренировочных данных
    pred = model.predict(X_test)  # Предсказываем результат на тестовой выборке

    # Вычисляем метрики качества модели и сохраняем их в словарь
    metrics = {
        "Accuracy": accuracy_score(Y_test, pred),        # Доля правильных предсказаний
        "Precision": precision_score(Y_test, pred),      # Точность — насколько предсказанные "1" действительно "1"
        "Recall": recall_score(Y_test, pred),            # Полнота — сколько реальных "1" модель нашла
        "F1 Score": f1_score(Y_test, pred)               # Сбалансированная мера точности и полноты
    }
    return metrics  # Возвращаем словарь с метриками

# Словарь, где хранятся модели, которые мы хотим сравнить
models = {
    "Random Forest": RandomForestClassifier(random_state=42),  # Модель случайного леса
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)  # Модель градиентного бустинга
}

# Цикл по всем моделям: обучаем каждую и выводим метрики
for name, model in models.items():
    metrics = train_and_evaluate(model, X_train, Y_train, X_test, Y_test)  # Получаем метрики для модели
    print(f"{name} Metrics:")  # Выводим название модели
    for metric, value in metrics.items():  # Перебираем и выводим каждую метрику
        print(f"{metric}: {value:.2f}")
    print()  # Пустая строка для отделения между моделями
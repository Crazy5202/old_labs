from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import pandas as pd
from typing import Callable

def remove_score_suffix(name):
    if name.endswith("_score"):
        return name[:-6]
    return name

def feature_selection_rfecv(
    data: pd.DataFrame, 
    target_column: str, 
    min_features: int, 
    metric: Callable,
    test_size: float = 0.2, 
    random_state: int = 42,
    selector_step: int = 10
) -> list[str]:
    """
    Отбор фичей с использованием RFECV.

    :param data: DataFrame с данными.
    :param target_column: Название колонки с целевой переменной.
    :param min_features: Количество фичей, которые нужно оставить.
    :param metric: Функция метрики,
    :param test_size: Доля тестовых данных при разделении.
    :param random_state: Начальное значение для генератора случайных чисел.
    :param selector_step: Сколько параметров убирается за шаг
    :return: Список оставшихся фичей.
    """
    # Разделение данных на X и y
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Инициализация модели и RFECV
    model = RandomForestClassifier(random_state=random_state)
    shap_elimination = RFECV(
        estimator=model,
        step=selector_step,  # Удаляем по одной фиче за итерацию
        min_features_to_select=min_features,
        cv=3,  # Кросс-валидация
        scoring=remove_score_suffix(metric.__name__),  # Метрика для оценки
    )
    
    shap_elimination.fit(X_train, y_train)
    
    # Список оставшихся фичей
    selected_features = X_train.columns[shap_elimination.support_]
    
    return selected_features
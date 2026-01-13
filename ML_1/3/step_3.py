from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd
from typing import Callable

def train_and_select_model(
    data: pd.DataFrame, 
    target_column: str,
    metric: Callable,
    test_size: float = 0.2, 
    random_state: int = 42,
    cv: int = 5
) -> object:
    """
    Тренировка моделей и выбор лучшей.

    :param data: DataFrame с данными.
    :param target_column: Название целевой колонки.
    :param test_size: Доля тестовой выборки.
    :param random_state: Начальное значение для генератора случайных чисел.
    :return: Название лучшей модели и её объект.
    """
    # Разделение данных
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Список моделей
    models = {
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
        "LightGBM": LGBMClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state)
    }
    
    # Оценка моделей
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metric(y_test, y_pred)
        model_scores[name] = (score, model)
    
    # Выбор лучшей модели
    best_model_name = max(model_scores, key=lambda x: model_scores[x][0])
    best_model = model_scores[best_model_name][1]
    
    print(f"Лучшая модель: {best_model_name} с значением метрики {model_scores[best_model_name][0]}")
    return best_model
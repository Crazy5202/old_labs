import pandas as pd
from typing import Callable

def evaluate_model(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    metric: Callable
) -> pd.DataFrame:
    """
    Оценивает модель, выводит метрику и возвращает датафрейм с результатами.

    :param model: Обученная модель с методами predict.
    :param X: Признаки для тестирования.
    :param y: Реальные значения целевой переменной.
    :param metric: Метрика для оценки модели.
    :return: DataFrame с признаками, предсказаниями, реальными значениями и значением метрики.
    """
    # Предсказания
    y_pred = model.predict(X)
    
    # Вычисление метрики
    metric_value = metric(y, y_pred)
    #print(f"Значение метрики ({metric.__name__}): {metric_value}")
    
    # Формирование результата
    results = {
        "X": X,
        "y_real": y,
        "y_pred": y_pred,
        "score": metric_value
    }
    
    return results
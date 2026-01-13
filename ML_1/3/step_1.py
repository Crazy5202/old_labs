import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(
        file_path: str, 
        target_column: str, 
        date_column: str = None, 
        drop_columns: list[str] = None, 
        fill_median: bool = False, 
        drop_empty: bool = False, 
        encode_objects: bool = False
) -> pd.DataFrame:
    """
    Загрузка данных из CSV, удаление ненужных колонок и создание фичей из даты.

    :param file_path: Путь к CSV файлу.
    :param drop_columns: Список колонок, которые нужно удалить.
    :param date_column: Название колонки с датами для извлечения фичей (опционально).
    :param target_column: Колонка таргета, которая энкодится
    :param fill_median: Включает заполнение пропусков среди численных столбцов средним
    :param drop_empty: Включает удаление пустых строк
    :param encode_objects: Включает энкодинг всех текстовых/объектных колонок

    :return: Обработанный DataFrame.
    """
    # Считывание данных
    df = pd.read_csv(file_path)

    # Обработка колонки с датой
    if date_column:
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            # Извлечение фичей из даты
            df['year'] = df[date_column].dt.year
            df['month'] = df[date_column].dt.month
            df['day'] = df[date_column].dt.day
            df['day_of_week'] = df[date_column].dt.dayofweek
            df['is_weekend'] = df[date_column].dt.dayofweek >= 5
            df['quarter'] = df[date_column].dt.quarter
            df.drop(columns=[date_column], inplace=True)
        else:
            raise ValueError(f"Колонка '{date_column}' не найдена в данных.")

    # Удаление ненужных колонок
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')

    # Заполнение медианным значением
    if fill_median:
        df.fillna(df.median(numeric_only=True), inplace = True)

    # Удаление строк с пропусками
    if drop_empty:
        df.dropna(inplace=True)

    # Энкодинг целевой колонки
    if target_column:
        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])

    # label-encoding
    if encode_objects:
        text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        df = pd.get_dummies(df, columns=text_columns)
    
    # Возврат обработанного DataFrame
    return df


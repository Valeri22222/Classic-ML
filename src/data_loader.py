"""Модуль для загрузки данных."""

import os
import pandas as pd
from pathlib import Path


def get_project_root():
    """Возвращает корневую папку проекта."""
    return Path(__file__).parent.parent


def load_raw_data(filename='Данные_для_курсовои__Классическое_МО.xlsx'):
    """Загружает исходные данные."""
    data_path = get_project_root() / 'data' / 'raw' / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"Файл не найден: {data_path}")
    
    return pd.read_excel(data_path)


def load_cleaned_data(filename='df_cleaned.csv'):
    """Загружает очищенные данные."""
    data_path = get_project_root() / 'data' / 'processed' / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"Файл не найден: {data_path}")
    
    return pd.read_csv(data_path)


"""
src/__init__.py
===============

Пакет с переиспользуемыми модулями для проекта по прогнозированию
биологической активности химических соединений.

Структураs пакета:
├── data_loader.py     — загрузка и сохранение данных
├── preprocess.py      — предобработка данных
├── models.py          — модели и гиперпараметры
├── evaluation.py      — метрики оценки
└── utils.py           — вспомогательные функции

Пример использования в ноутбуке:
    from src import load_raw_data, get_regression_models
    
    df = load_raw_data()
    models = get_regression_models()
"""


# Импорты для удобного доступа

# data_loader
from .data_loader import (
    load_raw_data,
    load_cleaned_data
)

# preprocess
from .preprocess import (
    prepare_data_for_modeling
)

# models
from .models import (
    get_regression_models,
    get_classification_models,
    get_param_grids
)

# evaluation
from .evaluation import (
    evaluate_regression,
    evaluate_classification
)

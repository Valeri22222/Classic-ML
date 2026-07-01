"""Модуль для определения и обучения моделей."""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier


def get_regression_models(random_state=42):
    """Возвращает словарь моделей регрессии."""
    return {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=random_state),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state, verbosity=0)
    }


def get_classification_models(random_state=42):
    """Возвращает словарь моделей классификации."""
    return {
        'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(random_state=random_state, verbosity=0, use_label_encoder=False)
    }


def get_param_grids():
    """Возвращает сетки параметров для оптимизации."""
    return {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        },
        'DecisionTree': {
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        }
    }
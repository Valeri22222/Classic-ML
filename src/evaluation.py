"""Модуль для оценки моделей."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)


def evaluate_regression(y_true, y_pred):
    """Оценивает модель регрессии."""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def evaluate_classification(y_true, y_pred, y_proba=None):
    """Оценивает модель классификации."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Добавляем AUC, если есть вероятности
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics['AUC'] = roc_auc_score(y_true, y_proba[:, 1])
    
    return metrics


"""Модуль для предобработки данных."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data_for_modeling(df, reg_targets, clf_targets, test_size=0.2, random_state=42):
    """Подготавливает данные для моделирования."""
    X = df.drop(columns=reg_targets + clf_targets + ['IC50, mM', 'CC50, mM', 'SI'])
    y_reg = df[reg_targets]
    y_clf = df[clf_targets]
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
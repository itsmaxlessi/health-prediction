"""
Класс для обучения модели
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import logging
from pathlib import Path
from typing import Optional

from ..preprocessing.data_preprocessor import DataPreprocessor
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Класс для обучения модели машинного обучения
    """
    
    def __init__(self, random_state: int = 42):
        """
        Инициализация тренера модели
        
        Args:
            random_state: Seed для воспроизводимости
        """
        self.random_state = random_state
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.feature_columns = None
        
    def train(
        self, 
        train_df: pd.DataFrame,
        target_column: str = "Heart Attack Risk (Binary)",
        test_size: float = 0.2
    ) -> dict:
        """
        Обучение модели
        
        Args:
            train_df: Обучающий датафрейм
            target_column: Название целевой колонки
            test_size: Размер тестовой выборки
        
        Returns:
            Словарь с метриками
        """
        logger.info("Начало обучения модели")
        
        # Предобработка данных
        X = self.preprocessor.fit_transform(train_df, target_column)
        y = train_df[target_column]
        
        # Разделение на train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Инициализация модели
        self.model = CatBoostClassifier(
            random_state=self.random_state,
            verbose=100,
            eval_metric='F1'
        )
        
        # Обучение
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50
        )
        
        # Предсказания и метрики
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'f1_score': f1_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        logger.info(f"Метрики на валидации: {metrics}")
        
        self.feature_columns = self.preprocessor.feature_columns
        
        return metrics
    
    def save_model(self, path: Optional[Path] = None):
        """
        Сохранение модели
        
        Args:
            path: Путь для сохранения (по умолчанию из Config)
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        if path is None:
            path = Config.MODEL_PATH
        
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"Модель сохранена в {path}")

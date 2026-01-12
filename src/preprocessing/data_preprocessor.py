"""
Класс для предобработки данных
"""
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс для предобработки данных перед обучением модели
    """
    
    def __init__(self):
        """Инициализация препроцессора"""
        self.scaler = StandardScaler()
        self.feature_columns: Optional[List[str]] = None
        self.columns_to_drop: List[str] = []
        
    def fit(self, df: pd.DataFrame, target_column: str = "Heart Attack Risk (Binary)") -> 'DataPreprocessor':
        """
        Обучение препроцессора на данных
        
        Args:
            df: Датафрейм с данными
            target_column: Название целевой колонки
        
        Returns:
            self
        """
        logger.info("Начало обучения препроцессора")
        
        # Определяем колонки признаков
        self.feature_columns = [
            col for col in df.columns 
            if col not in [target_column, 'id', 'Unnamed: 0']
        ]
        
        # TODO: Добавить логику удаления бесполезных признаков
        # TODO: Добавить обработку выбросов
        # TODO: Добавить обработку пропусков
        
        logger.info(f"Определено {len(self.feature_columns)} признаков")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение трансформаций к данным
        
        Args:
            df: Датафрейм для трансформации
        
        Returns:
            Трансформированный датафрейм
        """
        logger.info("Применение трансформаций")
        
        df_processed = df.copy()
        
        # TODO: Реализовать трансформации
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = "Heart Attack Risk (Binary)") -> pd.DataFrame:
        """
        Обучение и применение трансформаций
        
        Args:
            df: Датафрейм с данными
            target_column: Название целевой колонки
        
        Returns:
            Трансформированный датафрейм
        """
        return self.fit(df, target_column).transform(df)

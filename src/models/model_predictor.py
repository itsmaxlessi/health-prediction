"""
Класс для предсказаний модели
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from pathlib import Path
import logging

from ..preprocessing.data_preprocessor import DataPreprocessor
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Класс для загрузки модели и выполнения предсказаний
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Инициализация предиктора
        
        Args:
            model_path: Путь к сохраненной модели
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.model = None
        self.preprocessor = DataPreprocessor()
        
    def load_model(self):
        """Загрузка модели из файла"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))
        logger.info(f"Модель загружена из {self.model_path}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполнение предсказаний
        
        Args:
            df: Датафрейм с данными для предсказания
        
        Returns:
            Датафрейм с предсказаниями (id, prediction)
        """
        if self.model is None:
            self.load_model()
        
        # Предобработка
        X = self.preprocessor.transform(df)
        
        # Предсказания
        predictions = self.model.predict(X)
        
        # Формирование результата
        result = pd.DataFrame({
            'id': df['id'],
            'prediction': predictions
        })
        
        return result

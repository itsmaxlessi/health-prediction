"""
Класс для предсказаний модели V2
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

from ..preprocessing.preprocessor import DataPreprocessor
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Класс для загрузки модели V2 и выполнения предсказаний.
    Использует оптимальный порог (0.40) для улучшения F1-score.
    """
    
    def __init__(self, model_path: Optional[Path] = None, threshold_path: Optional[Path] = None):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к сохраненной модели
            threshold_path: Путь к файлу с порогом
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.threshold_path = threshold_path or Config.THRESHOLD_PATH
        self.model = None
        self.threshold = Config.DEFAULT_THRESHOLD
        self.preprocessor = DataPreprocessor()
        self._is_loaded = False
        
    def load_model(self) -> None:
        """Загрузка модели и порога из файлов."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {self.model_path}. "
                "Запустите ноутбук 02_model_v2.ipynb для сохранения модели."
            )
        
        self.model = joblib.load(self.model_path)
        logger.info(f"Модель загружена из {self.model_path}")
        
        # Загружаем порог
        if self.threshold_path.exists():
            self.threshold = joblib.load(self.threshold_path)
            logger.info(f"Порог загружен: {self.threshold}")
        else:
            self.threshold = Config.DEFAULT_THRESHOLD
            logger.warning(f"Файл порога не найден, используем по умолчанию: {self.threshold}")
        
        self._is_loaded = True
    
    def _ensure_loaded(self) -> None:
        """Проверка что модель загружена."""
        if not self._is_loaded or self.model is None:
            self.load_model()
    
    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание для одного пациента с оптимальным порогом.
        
        Args:
            patient_data: Словарь с данными пациента
        
        Returns:
            Словарь с результатом предсказания
        """
        self._ensure_loaded()
        
        # Предобработка (включая Feature Engineering)
        X = self.preprocessor.preprocess_single(patient_data)
        
        # Получаем вероятность
        probability = float(self.model.predict_proba(X)[0][1])
        
        # Применяем оптимальный порог (0.40)
        prediction = 1 if probability >= self.threshold else 0
        
        # Интерпретация
        risk_level = "HIGH" if prediction == 1 else "LOW"
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "risk_percentage": f"{probability * 100:.1f}%",
            "threshold_used": self.threshold
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполнение предсказаний для DataFrame.
        
        Args:
            df: Датафрейм с данными для предсказания
        
        Returns:
            Датафрейм с предсказаниями
        """
        self._ensure_loaded()
        
        # Сохраняем id если есть
        ids = df['id'].values if 'id' in df.columns else range(len(df))
        
        # Предобработка (включая Feature Engineering)
        X = self.preprocessor.transform(df)
        
        # Получаем вероятности
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Применяем оптимальный порог
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Формирование результата
        result = pd.DataFrame({
            'id': ids,
            'prediction': predictions,
            'probability': probabilities.round(4)
        })
        
        logger.info(f"Выполнено предсказаний: {len(result)}")
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        self._ensure_loaded()
        
        info = {
            "model_type": type(self.model).__name__,
            "model_path": str(self.model_path),
            "threshold": self.threshold,
            "is_loaded": self._is_loaded,
            "version": "V2 (with Feature Engineering)"
        }
        
        # Добавляем параметры модели если доступны
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            info["n_estimators"] = params.get("n_estimators")
            info["max_depth"] = params.get("max_depth")
        
        return info

"""
Класс для предобработки данных (V2 с Feature Engineering)
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from pathlib import Path

from ..utils.config import Config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс для предобработки данных перед предсказанием.
    Включает Feature Engineering из Model V2.
    """
    
    # Список БАЗОВЫХ признаков (до feature engineering)
    BASE_FEATURES = [
        'Age', 'Cholesterol', 'Heart rate', 'Diabetes', 'Family History',
        'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
        'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
        'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
        'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Blood sugar',
        'CK-MB', 'Troponin', 'Gender', 'Systolic blood pressure',
        'Diastolic blood pressure'
    ]
    
    # Значения для заполнения пропусков (моды из train)
    DEFAULT_FILL_VALUES = {
        'Diabetes': 1.0,
        'Family History': 0.0,
        'Smoking': 1.0,
        'Obesity': 0.0,
        'Alcohol Consumption': 1.0,
        'Previous Heart Problems': 1.0,
        'Medication Use': 0.0,
        'Stress Level': 5.0,
        'Physical Activity Days Per Week': 3.0
    }
    
    def __init__(self):
        """Инициализация препроцессора"""
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['Female', 'Male'])  # Female=0, Male=1
        self.fill_values = self.DEFAULT_FILL_VALUES.copy()
        
        # Попробуем загрузить сохранённые значения
        try:
            if Config.FILL_VALUES_PATH.exists():
                self.fill_values = joblib.load(Config.FILL_VALUES_PATH)
                logger.info("Загружены значения заполнения из файла")
        except Exception as e:
            logger.warning(f"Не удалось загрузить fill_values: {e}")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering из Model V2.
        Создаёт 17 новых признаков.
        """
        df = df.copy()
        
        # === 1. Комбинированные факторы риска ===
        df['Lifestyle_Risk'] = (
            df['Smoking'] + 
            df['Obesity'] + 
            df['Alcohol Consumption'] + 
            (1 - df['Physical Activity Days Per Week'] / 7)
        )
        
        df['Medical_Risk'] = (
            df['Diabetes'] + 
            df['Family History'] + 
            df['Previous Heart Problems']
        )
        
        df['Total_Risk_Score'] = df['Lifestyle_Risk'] + df['Medical_Risk']
        
        # === 2. Взаимодействия признаков ===
        df['Age_BMI'] = df['Age'] * df['BMI']
        df['Age_Cholesterol'] = df['Age'] * df['Cholesterol']
        df['Lipid_Total'] = df['Cholesterol'] + df['Triglycerides']
        
        # === 3. Соотношения давления ===
        df['Pulse_Pressure'] = df['Systolic blood pressure'] - df['Diastolic blood pressure']
        df['Mean_Arterial_Pressure'] = (
            df['Diastolic blood pressure'] + 
            (df['Systolic blood pressure'] - df['Diastolic blood pressure']) / 3
        )
        
        # === 4. Биомаркеры ===
        df['Cardiac_Biomarkers'] = df['CK-MB'] + df['Troponin']
        
        # === 5. Образ жизни ===
        df['Activity_Balance'] = df['Exercise Hours Per Week'] - df['Sedentary Hours Per Day']
        df['Sleep_Quality'] = 1 - np.abs(df['Sleep Hours Per Day'] - 0.5) * 2
        
        # === 6. Полиномиальные признаки ===
        df['Age_squared'] = df['Age'] ** 2
        df['BMI_squared'] = df['BMI'] ** 2
        df['Cholesterol_squared'] = df['Cholesterol'] ** 2
        
        # === 7. Категориальные взаимодействия ===
        df['Smoking_Diabetes'] = df['Smoking'] * df['Diabetes']
        df['Smoking_FamilyHistory'] = df['Smoking'] * df['Family History']
        df['Obesity_Diabetes'] = df['Obesity'] * df['Diabetes']
        df['Stress_Sedentary'] = df['Stress Level'] * df['Sedentary Hours Per Day']
        
        return df
    
    def preprocess_single(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Предобработка данных одного пациента.
        
        Args:
            data: Словарь с данными пациента
        
        Returns:
            DataFrame готовый для предсказания
        """
        df = pd.DataFrame([data])
        return self.transform(df)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение трансформаций к данным.
        
        Args:
            df: Датафрейм для трансформации
        
        Returns:
            Трансформированный датафрейм
        """
        logger.info("Начало предобработки данных")
        df_processed = df.copy()
        
        # 1. Удаление служебных колонок
        for col in Config.COLUMNS_TO_DROP:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
                logger.debug(f"Удалена колонка: {col}")
        
        # 2. Исправление Gender (0.0/1.0 -> Female/Male -> 0/1)
        if 'Gender' in df_processed.columns:
            df_processed['Gender'] = df_processed['Gender'].replace({
                '0.0': 'Female', '1.0': 'Male',
                0.0: 'Female', 1.0: 'Male',
                '0': 'Female', '1': 'Male',
                0: 'Female', 1: 'Male'
            })
            df_processed['Gender'] = df_processed['Gender'].map(
                {'Female': 0, 'Male': 1}
            ).fillna(df_processed['Gender'])
            if df_processed['Gender'].dtype == 'object':
                df_processed['Gender'] = self.label_encoder.transform(
                    df_processed['Gender']
                )
        
        # 3. Заполнение пропусков модой
        for col, value in self.fill_values.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(value)
        
        # 4. Убеждаемся, что все базовые колонки присутствуют
        for col in self.BASE_FEATURES:
            if col not in df_processed.columns:
                logger.warning(f"Отсутствует колонка {col}, добавляем с значением 0")
                df_processed[col] = 0
        
        # 5. Выбираем только базовые колонки
        df_processed = df_processed[self.BASE_FEATURES]
        
        # 6. Преобразуем в числовой тип
        df_processed = df_processed.astype(float)
        
        # 7. Feature Engineering (V2)
        df_processed = self._create_features(df_processed)
        
        logger.info(f"Предобработка завершена. Shape: {df_processed.shape}")
        return df_processed

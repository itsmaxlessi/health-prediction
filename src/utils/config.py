"""
Конфигурация проекта
"""
import os
from pathlib import Path


class Config:
    """Класс для хранения конфигурации проекта"""
    
    # Пути к данным
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    
    # Файлы данных
    TRAIN_FILE = RAW_DATA_DIR / "heart_train.csv"
    TEST_FILE = RAW_DATA_DIR / "heart_test.csv"
    
    # Модель V2 (Random Forest с оптимальным порогом)
    MODEL_PATH = MODELS_DIR / "model_v2.joblib"
    THRESHOLD_PATH = MODELS_DIR / "threshold_v2.joblib"
    FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
    FILL_VALUES_PATH = MODELS_DIR / "fill_values.joblib"
    
    # Настройки модели
    RANDOM_STATE = 42
    DEFAULT_THRESHOLD = 0.40  # Оптимальный порог из V2
    
    # FastAPI
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Целевая переменная
    TARGET_COLUMN = "Heart Attack Risk (Binary)"
    
    # Колонки для удаления
    COLUMNS_TO_DROP = ["Unnamed: 0", "id"]
    
    # Колонки с пропусками (для заполнения модой)
    COLUMNS_WITH_MISSING = [
        "Diabetes", "Family History", "Smoking", "Obesity",
        "Alcohol Consumption", "Previous Heart Problems",
        "Medication Use", "Stress Level", "Physical Activity Days Per Week"
    ]
    
    @classmethod
    def create_directories(cls):
        """Создает необходимые директории"""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(exist_ok=True)

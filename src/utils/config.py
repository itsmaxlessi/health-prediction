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
    TEST_FILE = RAW_DATA_DIR / "model_predictions.csv"
    
    # Модель
    MODEL_PATH = MODELS_DIR / "heart_attack_model.cbm"
    
    # Настройки модели
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # FastAPI
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    @classmethod
    def create_directories(cls):
        """Создает необходимые директории"""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(exist_ok=True)

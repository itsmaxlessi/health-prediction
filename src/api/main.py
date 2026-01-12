"""
FastAPI приложение для предсказания риска сердечных приступов
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import logging
from pathlib import Path

from ..models.model_predictor import ModelPredictor
from ..utils.config import Config
from ..utils.logger import setup_logger

# Настройка логирования
logger = setup_logger()

# Инициализация FastAPI
app = FastAPI(
    title="Health Prediction API",
    description="API для предсказания риска сердечных приступов",
    version="0.1.0"
)

# Глобальный предиктор
predictor = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    global predictor
    try:
        predictor = ModelPredictor()
        predictor.load_model()
        logger.info("API готов к работе")
    except FileNotFoundError:
        logger.warning("Модель не найдена. Обучение модели перед использованием API.")


class PredictionRequest(BaseModel):
    """Модель запроса для предсказания"""
    file_path: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с документацией"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Health Prediction API</h1>
        <p>API для предсказания риска сердечных приступов</p>
        
        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Загрузить CSV файл и получить предсказания</p>
            <p>Используйте: <code>curl -X POST "http://localhost:8000/predict" -F "file=@test.csv"</code></p>
        </div>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Проверка статуса API</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /docs</h3>
            <p>Интерактивная документация Swagger</p>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Предсказание на основе загруженного CSV файла
    
    Args:
        file: CSV файл с тестовыми данными
    
    Returns:
        JSON с предсказаниями
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        # Чтение CSV файла
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Выполнение предсказаний
        predictions = predictor.predict(df)
        
        # Конвертация в словарь для JSON
        result = predictions.to_dict(orient='records')
        
        logger.info(f"Выполнено предсказаний: {len(result)}")
        
        return JSONResponse(content={"predictions": result})
    
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

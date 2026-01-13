"""
FastAPI приложение для предсказания риска сердечных приступов (Model V2)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import logging
from pathlib import Path
import io

from ..prediction.predictor import ModelPredictor
from ..utils.config import Config
from ..utils.logger import setup_logger

# Настройка логирования
logger = setup_logger()

# Инициализация FastAPI
app = FastAPI(
    title="Heart Attack Risk Prediction API (V2)",
    description="API для предсказания риска сердечных приступов. Использует Model V2 с Feature Engineering и оптимальным порогом 0.40",
    version="2.0.0"
)

# Глобальный предиктор
predictor: Optional[ModelPredictor] = None


class PatientData(BaseModel):
    """Модель данных пациента для предсказания"""
    
    # Числовые признаки (0-1 нормализованные)
    Age: float = Field(..., ge=0, le=1, description="Возраст (нормализованный 0-1)")
    Cholesterol: float = Field(..., ge=0, le=1, description="Холестерин (нормализованный 0-1)")
    Heart_rate: float = Field(..., ge=0, le=1, description="Пульс (нормализованный 0-1)")
    Exercise_Hours_Per_Week: float = Field(..., ge=0, le=1, description="Часы упражнений в неделю (нормализованный 0-1)")
    Sedentary_Hours_Per_Day: float = Field(..., ge=0, le=1, description="Сидячие часы в день (нормализованный 0-1)")
    Income: float = Field(..., ge=0, le=1, description="Доход (нормализованный 0-1)")
    BMI: float = Field(..., ge=0, le=1, description="Индекс массы тела (нормализованный 0-1)")
    Triglycerides: float = Field(..., ge=0, le=1, description="Триглицериды (нормализованный 0-1)")
    Sleep_Hours_Per_Day: float = Field(..., ge=0, le=1, description="Часы сна в день (нормализованный 0-1)")
    Blood_sugar: float = Field(..., ge=0, le=1, description="Сахар в крови (нормализованный 0-1)")
    CK_MB: float = Field(..., ge=0, le=1, description="CK-MB (нормализованный 0-1)")
    Troponin: float = Field(..., ge=0, le=1, description="Тропонин (нормализованный 0-1)")
    Systolic_blood_pressure: float = Field(..., ge=0, le=1, description="Систолическое давление (нормализованный 0-1)")
    Diastolic_blood_pressure: float = Field(..., ge=0, le=1, description="Диастолическое давление (нормализованный 0-1)")
    
    # Бинарные признаки (0 или 1)
    Diabetes: int = Field(..., ge=0, le=1, description="Диабет (0=нет, 1=да)")
    Family_History: int = Field(..., ge=0, le=1, description="Семейная история (0=нет, 1=да)")
    Smoking: int = Field(..., ge=0, le=1, description="Курение (0=нет, 1=да)")
    Obesity: int = Field(..., ge=0, le=1, description="Ожирение (0=нет, 1=да)")
    Alcohol_Consumption: int = Field(..., ge=0, le=1, description="Употребление алкоголя (0=нет, 1=да)")
    Previous_Heart_Problems: int = Field(..., ge=0, le=1, description="Предыдущие проблемы с сердцем (0=нет, 1=да)")
    Medication_Use: int = Field(..., ge=0, le=1, description="Использование лекарств (0=нет, 1=да)")
    
    # Категориальные признаки
    Gender: str = Field(..., description="Пол (Male/Female)")
    Diet: int = Field(..., ge=0, le=2, description="Диета (0=плохая, 1=средняя, 2=хорошая)")
    Stress_Level: int = Field(..., ge=1, le=10, description="Уровень стресса (1-10)")
    Physical_Activity_Days_Per_Week: int = Field(..., ge=0, le=7, description="Дни физической активности в неделю (0-7)")
    
    def to_model_dict(self) -> dict:
        """Преобразование в словарь для модели"""
        return {
            'Age': self.Age,
            'Cholesterol': self.Cholesterol,
            'Heart rate': self.Heart_rate,
            'Diabetes': self.Diabetes,
            'Family History': self.Family_History,
            'Smoking': self.Smoking,
            'Obesity': self.Obesity,
            'Alcohol Consumption': self.Alcohol_Consumption,
            'Exercise Hours Per Week': self.Exercise_Hours_Per_Week,
            'Diet': self.Diet,
            'Previous Heart Problems': self.Previous_Heart_Problems,
            'Medication Use': self.Medication_Use,
            'Stress Level': self.Stress_Level,
            'Sedentary Hours Per Day': self.Sedentary_Hours_Per_Day,
            'Income': self.Income,
            'BMI': self.BMI,
            'Triglycerides': self.Triglycerides,
            'Physical Activity Days Per Week': self.Physical_Activity_Days_Per_Week,
            'Sleep Hours Per Day': self.Sleep_Hours_Per_Day,
            'Blood sugar': self.Blood_sugar,
            'CK-MB': self.CK_MB,
            'Troponin': self.Troponin,
            'Gender': self.Gender,
            'Systolic blood pressure': self.Systolic_blood_pressure,
            'Diastolic blood pressure': self.Diastolic_blood_pressure,
        }


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    global predictor
    try:
        predictor = ModelPredictor()
        predictor.load_model()
        logger.info(f"Model V2 загружена. Порог: {predictor.threshold}")
        logger.info("API готов к работе")
    except FileNotFoundError as e:
        logger.warning(f"Модель не найдена: {e}")
        logger.warning("Запустите ноутбук 02_model_v2.ipynb для сохранения модели.")


# HTML страница с загрузкой CSV
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Attack Risk Prediction V2</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 { 
            color: #333; 
            text-align: center; 
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .version-badge {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .upload-section h2 {
            margin-top: 0;
            color: #495057;
        }
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            margin: 20px 0;
        }
        .file-input-wrapper input[type="file"] {
            font-size: 16px;
            padding: 15px;
            border: 3px dashed #667eea;
            border-radius: 10px;
            background: white;
            cursor: pointer;
            width: 100%;
            max-width: 400px;
        }
        .file-input-wrapper input[type="file"]:hover {
            border-color: #764ba2;
            background: #f0f0ff;
        }
        button { 
            padding: 15px 40px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border: none; 
            border-radius: 10px; 
            font-size: 18px; 
            font-weight: 600;
            cursor: pointer; 
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 10px;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        button.secondary {
            background: #6c757d;
        }
        button.secondary:hover {
            background: #5a6268;
        }
        .results-section {
            display: none;
            margin-top: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-card.high-risk {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        .stat-card.low-risk {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .risk-high {
            color: #dc3545;
            font-weight: bold;
        }
        .risk-low {
            color: #28a745;
            font-weight: bold;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #f8d7da;
            border: 2px solid #dc3545;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        .info-box {
            background: #e7f3ff;
            border: 1px solid #b6d4fe;
            color: #084298;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .model-info {
            background: #e9ecef;
            padding: 15px;
            border-radius: 10px;
            margin-top: 30px;
        }
        .model-info h3 {
            margin-top: 0;
        }
        .table-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Heart Attack Risk Prediction <span class="version-badge">V2</span></h1>
        <p class="subtitle">Загрузите CSV файл с данными пациентов для анализа риска</p>
        
        <div class="upload-section">
            <h2>Загрузка данных</h2>
            
            <div class="info-box">
                <strong>Формат файла:</strong> CSV с колонками из тренировочного датасета (heart_train.csv)<br>
                Данные должны быть нормализованы (0-1) как в исходном датасете.
            </div>
            
            <div class="file-input-wrapper">
                <input type="file" id="csvFile" accept=".csv">
            </div>
            
            <div>
                <button onclick="uploadFile()" id="uploadBtn">Анализировать</button>
                <button onclick="downloadSample()" class="secondary">Скачать пример</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Обработка данных...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results-section" id="results">
            <h2>Результаты анализа</h2>
            
            <div class="stats-grid">
                <div class="stat-card" id="totalCard">
                    <div class="stat-number" id="totalCount">0</div>
                    <div class="stat-label">Всего записей</div>
                </div>
                <div class="stat-card high-risk" id="highRiskCard">
                    <div class="stat-number" id="highRiskCount">0</div>
                    <div class="stat-label">Высокий риск</div>
                </div>
                <div class="stat-card low-risk" id="lowRiskCard">
                    <div class="stat-number" id="lowRiskCount">0</div>
                    <div class="stat-label">Низкий риск</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="avgProb">0%</div>
                    <div class="stat-label">Средняя вероятность</div>
                </div>
            </div>
            
            <div style="margin-bottom: 15px;">
                <button onclick="downloadResults()" class="secondary">Скачать результаты (CSV)</button>
            </div>
            
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Риск</th>
                            <th>Вероятность</th>
                        </tr>
                    </thead>
                    <tbody id="resultsTable">
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="model-info">
            <h3>О модели</h3>
            <p><strong>Версия:</strong> V2 с Feature Engineering</p>
            <p><strong>Порог классификации:</strong> 0.40 (оптимизированный)</p>
            <p><strong>F1-score:</strong> 0.526</p>
            <p><strong>Алгоритм:</strong> Random Forest с 17 дополнительными признаками</p>
        </div>
    </div>

    <script>
        let lastResults = null;
        
        async function uploadFile() {
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('Пожалуйста, выберите CSV файл');
                return;
            }
            
            const uploadBtn = document.getElementById('uploadBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const error = document.getElementById('error');
            
            uploadBtn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';
            error.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict/csv', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    showError(data.detail || 'Ошибка при обработке файла');
                }
            } catch (err) {
                showError('Ошибка соединения: ' + err.message);
            } finally {
                uploadBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        function displayResults(data) {
            lastResults = data.predictions;
            
            const predictions = data.predictions;
            const total = predictions.length;
            const highRisk = predictions.filter(p => p.prediction === 1).length;
            const lowRisk = total - highRisk;
            const avgProb = predictions.reduce((sum, p) => sum + p.probability, 0) / total;
            
            document.getElementById('totalCount').textContent = total;
            document.getElementById('highRiskCount').textContent = highRisk;
            document.getElementById('lowRiskCount').textContent = lowRisk;
            document.getElementById('avgProb').textContent = (avgProb * 100).toFixed(1) + '%';
            
            const tbody = document.getElementById('resultsTable');
            tbody.innerHTML = '';
            
            predictions.forEach(p => {
                const row = document.createElement('tr');
                const riskClass = p.prediction === 1 ? 'risk-high' : 'risk-low';
                const riskText = p.prediction === 1 ? 'ВЫСОКИЙ' : 'НИЗКИЙ';
                
                row.innerHTML = `
                    <td>${p.id}</td>
                    <td class="${riskClass}">${riskText}</td>
                    <td>${(p.probability * 100).toFixed(1)}%</td>
                `;
                tbody.appendChild(row);
            });
            
            document.getElementById('results').style.display = 'block';
        }
        
        function showError(message) {
            const error = document.getElementById('error');
            error.textContent = message;
            error.style.display = 'block';
        }
        
        function downloadResults() {
            if (!lastResults) return;
            
            let csv = 'id,prediction,probability\\n';
            lastResults.forEach(p => {
                csv += `${p.id},${p.prediction},${p.probability}\\n`;
            });
            
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'predictions.csv';
            a.click();
            URL.revokeObjectURL(url);
        }
        
        function downloadSample() {
            window.location.href = '/sample';
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с формой загрузки CSV"""
    return HTML_PAGE


@app.get("/sample")
async def download_sample():
    """Скачать пример CSV файла (первые 10 строк из test)"""
    try:
        test_path = Config.TEST_FILE
        if test_path.exists():
            df = pd.read_csv(test_path).head(10)
            
            # Создаём CSV в памяти
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=sample_data.csv"}
            )
        else:
            raise HTTPException(status_code=404, detail="Test file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    model_loaded = predictor is not None and predictor._is_loaded
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": "V2",
        "threshold": predictor.threshold if predictor else None,
        "message": "Ready for predictions" if model_loaded else "Model not loaded"
    }


@app.get("/model/info")
async def model_info():
    """Информация о загруженной модели"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        return predictor.get_model_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/patient")
async def predict_patient(patient: PatientData):
    """
    Предсказание риска для одного пациента.
    Использует Model V2 с оптимальным порогом 0.40.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        result = predictor.predict_single(patient.to_model_dict())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Предсказание для CSV файла.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        predictions = predictor.predict(df)
        result = predictions.to_dict(orient='records')
        
        logger.info(f"Processed {len(result)} predictions from CSV")
        
        return JSONResponse(content={"predictions": result, "count": len(result)})
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

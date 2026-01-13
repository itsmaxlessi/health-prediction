# API Documentation

> REST API для предсказания риска сердечного приступа

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### `GET /` — Веб-интерфейс

Главная страница с формой загрузки CSV файла.

**Response:** HTML страница

---

### `POST /predict/csv` — Предсказание для CSV файла

Загрузка CSV файла с данными пациентов и получение предсказаний.

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV файл с полем `file`

**Response:**
```json
{
  "predictions": [
    {
      "id": 0,
      "prediction": 1,
      "probability": 0.4523
    }
  ],
  "count": 100
}
```

**Пример (curl):**
```bash
curl -X POST "http://localhost:8000/predict/csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/raw/heart_test.csv"
```

---

### `POST /predict/patient` — Предсказание для одного пациента

Предсказание риска для одного пациента по JSON данным.

**Request:**
- Content-Type: `application/json`
- Body: JSON объект с данными пациента

**Request Body Schema:**
```json
{
  "Age": 0.5,
  "Cholesterol": 0.6,
  "Heart_rate": 0.5,
  "Exercise_Hours_Per_Week": 0.3,
  "Sedentary_Hours_Per_Day": 0.4,
  "Income": 0.5,
  "BMI": 0.6,
  "Triglycerides": 0.5,
  "Sleep_Hours_Per_Day": 0.5,
  "Blood_sugar": 0.4,
  "CK_MB": 0.3,
  "Troponin": 0.2,
  "Systolic_blood_pressure": 0.5,
  "Diastolic_blood_pressure": 0.4,
  "Diabetes": 0,
  "Family_History": 1,
  "Smoking": 0,
  "Obesity": 0,
  "Alcohol_Consumption": 0,
  "Previous_Heart_Problems": 0,
  "Medication_Use": 0,
  "Gender": "Male",
  "Diet": 1,
  "Stress_Level": 5,
  "Physical_Activity_Days_Per_Week": 3
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.3245,
  "risk_level": "LOW",
  "risk_percentage": "32.4%",
  "threshold_used": 0.4
}
```

---

### `GET /sample` — Скачать пример данных

Скачивает пример CSV файла (первые 10 строк из тестовой выборки).

**Response:** CSV файл

---

### `GET /health` — Статус API

Проверка работоспособности API.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "V2",
  "threshold": 0.4,
  "message": "Ready for predictions"
}
```

---

### `GET /model/info` — Информация о модели

Подробная информация о загруженной модели.

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "model_path": "models/model_v2.joblib",
  "threshold": 0.4,
  "is_loaded": true,
  "version": "V2 (with Feature Engineering)",
  "n_estimators": 100,
  "max_depth": 10
}
```

---

### `GET /docs` — Swagger UI

Автоматически сгенерированная интерактивная документация API.

---

## Коды ответов

| Код | Описание |
|-----|----------|
| 200 | Успешный запрос |
| 400 | Неверный формат запроса |
| 404 | Ресурс не найден |
| 500 | Внутренняя ошибка сервера |
| 503 | Модель не загружена |

---

## Формат данных

### Числовые признаки (нормализованные 0-1)

| Поле | Описание |
|------|----------|
| `Age` | Возраст |
| `Cholesterol` | Холестерин |
| `Heart_rate` | Пульс |
| `BMI` | Индекс массы тела |
| `Income` | Доход |
| `Triglycerides` | Триглицериды |
| `Blood_sugar` | Сахар в крови |
| `CK_MB` | Креатинкиназа MB |
| `Troponin` | Тропонин |
| `Systolic_blood_pressure` | Систолическое давление |
| `Diastolic_blood_pressure` | Диастолическое давление |
| `Exercise_Hours_Per_Week` | Часы упражнений в неделю |
| `Sedentary_Hours_Per_Day` | Сидячие часы в день |
| `Sleep_Hours_Per_Day` | Часы сна в день |

### Бинарные признаки (0 или 1)

| Поле | Описание |
|------|----------|
| `Diabetes` | Диабет |
| `Family_History` | Семейная история заболеваний |
| `Smoking` | Курение |
| `Obesity` | Ожирение |
| `Alcohol_Consumption` | Употребление алкоголя |
| `Previous_Heart_Problems` | Предыдущие проблемы с сердцем |
| `Medication_Use` | Приём медикаментов |

### Категориальные признаки

| Поле | Значения |
|------|----------|
| `Gender` | `"Male"` / `"Female"` |
| `Diet` | `0` (плохая), `1` (средняя), `2` (хорошая) |
| `Stress_Level` | `1-10` |
| `Physical_Activity_Days_Per_Week` | `0-7` |

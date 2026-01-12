# Health Prediction Project

Проект для предсказания риска сердечных приступов на основе данных пациентов.

## Описание

Модель машинного обучения для бинарной классификации риска сердечных приступов с использованием CatBoost и FastAPI для предоставления предсказаний через REST API.

## Структура проекта

```
health-prediction/
├── data/
│   ├── raw/              # Исходные данные
│   └── processed/        # Обработанные данные
├── notebooks/            # Jupyter notebooks для EDA и экспериментов
├── src/                  # Исходный код приложения
│   ├── api/              # FastAPI приложение
│   ├── models/           # Модели ML
│   ├── preprocessing/     # Предобработка данных
│   └── utils/            # Утилиты
├── models/               # Сохраненные модели
├── tests/                # Тесты
├── docs/                 # Документация
├── requirements.txt      # Зависимости
└── README.md            # Этот файл
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/itsmaxlessi/health-prediction.git
cd health-prediction
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Обучение модели

См. Jupyter Notebook в папке `notebooks/`

### Запуск API

```bash
uvicorn src.api.main:app --reload
```

API будет доступен по адресу: http://localhost:8000

### Использование API

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@data/raw/heart_test.csv"
```

## Документация API

После запуска API доступна интерактивная документация:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Лицензия

MIT

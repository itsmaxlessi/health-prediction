# Heart Attack Risk Prediction

Проект машинного обучения для предсказания риска сердечного приступа.

## Результаты

| Метрика | Значение |
|---------|----------|
| **F1-score** | 0.526 |
| Порог классификации | 0.40 |
| Модель | Random Forest |

---

## Быстрый старт

### 1. Установка

```bash
# Клонирование репозитория
git clone https://github.com/username/health-prediction.git
cd health-prediction

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Запуск API

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Открыть в браузере

```
http://localhost:8000
```

---

## Структура проекта

```
health-prediction/
│
├── data/                           # Данные
│   ├── raw/                        # Исходные данные
│   │   ├── heart_train.csv         # Обучающая выборка (8685 записей)
│   │   └── heart_test.csv          # Тестовая выборка (966 записей)
│   └── processed/                  # Результаты
│       └── submission.csv          # Предсказания на тестовой выборке
│
├── docs/                           # Документация
│   ├── API.md                      # Документация REST API
│   ├── TASK.md                     # Техническое задание
│   └── DEVELOPMENT.md              # История разработки
│
├── models/                         # Сохранённые модели (.joblib)
│   ├── model_v2.joblib             # Обученная модель
│   ├── threshold_v2.joblib         # Оптимальный порог (0.40)
│   └── feature_engineering_v2.joblib
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_eda.ipynb                # EDA + базовая модель (F1=0.48)
│   └── 02_model_v2.ipynb           # Улучшенная модель (F1=0.526)
│
├── src/                            # Исходный код
│   ├── api/                        # FastAPI приложение
│   │   └── main.py                 # Эндпоинты и веб-интерфейс
│   ├── prediction/                 # Модуль предсказаний
│   │   └── predictor.py            # Класс ModelPredictor
│   ├── preprocessing/              # Предобработка данных
│   │   └── preprocessor.py         # Класс DataPreprocessor
│   └── utils/                      # Утилиты
│       ├── config.py               # Конфигурация
│       └── logger.py               # Логирование
│
├── tests/                          # Тесты
│
├── .gitignore
├── README.md                       # Этот файл
└── requirements.txt                # Python зависимости
```

---

## API

### Эндпоинты

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Веб-интерфейс (загрузка CSV) |
| `/predict/csv` | POST | Предсказание для CSV файла |
| `/predict/patient` | POST | Предсказание для одного пациента |
| `/sample` | GET | Скачать пример данных |
| `/health` | GET | Статус API |
| `/model/info` | GET | Информация о модели |
| `/docs` | GET | Swagger документация |

> Подробная документация: [docs/API.md](docs/API.md)

---

## Notebooks

### `01_eda.ipynb` — Исследование данных

- Загрузка и анализ данных
- Обработка пропусков и выбросов
- Корреляционный анализ
- Базовая модель: **F1 = 0.48**

### `02_model_v2.ipynb` — Улучшенная модель

- Feature Engineering (17 новых признаков)
- Сравнение моделей (RF, XGBoost, LightGBM)
- Оптимизация порога классификации
- Финальная модель: **F1 = 0.526**

---

## Технологии

- **Python 3.12**
- **FastAPI** — REST API
- **Scikit-learn** — машинное обучение
- **Pandas, NumPy** — обработка данных
- **XGBoost, LightGBM** — градиентный бустинг
- **Matplotlib, Seaborn** — визуализация

---

## Классы

### `ModelPredictor` (src/prediction/predictor.py)

Класс для загрузки модели и выполнения предсказаний.

```python
from src.prediction.predictor import ModelPredictor

predictor = ModelPredictor()
predictor.load_model()

# Предсказание для одного пациента
result = predictor.predict_single(patient_data)

# Предсказание для DataFrame
predictions = predictor.predict(df)
```

### `DataPreprocessor` (src/preprocessing/preprocessor.py)

Класс для предобработки данных с Feature Engineering.

```python
from src.preprocessing.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
X_processed = preprocessor.transform(df)
```

### `Config` (src/utils/config.py)

Конфигурация проекта (пути к файлам, параметры модели).

---

## Лицензия

MIT

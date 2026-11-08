# Rossmann Store Sales Forecasting Demo Pipeline

## 1. Архітектура системи
Проєкт реалізує багаторівневий ML-конвеєр для прогнозування щоденного виторгу 100 магазинів Rossmann відповідно до методичних вимог:
- **FastAPI ML Microservice (`api/`)**: автономний бекенд-сервіс, який володіє навчанням моделей, валідацією, персистентністю та інференсом. Надає синхронне тренування, розрахунок метрик, кривих навчання та важливості ознак.
- **Streamlit Frontend (`streamlit_app/`)**: клієнтський веб-застосунок (4 сторінки на Plotly: EDA, Навчання, Порівняння моделей, Прогнозування/Сценарне моделювання). Спілкується з мікросервісом **виключно через REST API** за допомогою `api_client.py`, не імпортуючи жодного коду ML.
- **Notebook Client (`notebooks/client_demo.ipynb`)**: Jupyter-клієнт для відтворення навчання та прогнозу через HTTP-запити.
- **Models & Preprocessing (`api/models/`, `api/features.py`)**:
  - `RandomForestRegressor`, `LGBMRegressor` (з early stopping), `KerasMLP` (з стандартизацією входу і цілі, Dense+ReLU+Dropout, MAE loss, Adam).
  - Спільний уніфікований інтерфейс (`fit`, `predict`, `save`, `load`).
  - Scikit-learn `ColumnTransformer` з OHE (`handle_unknown='ignore'`).
  - Суворе дотримання правил витоку даних: `Customers` не використовується у навчанні; `logMonthSales` перераховується строго з train; навчання проводиться лише на відкритих днях (`Open == 1`), для закритих днів (`Open == 0`) прогнозується `0.0`.

```
app_sales_forecasting/
├── api/                   # FastAPI мікросервіс
│   ├── main.py            # Маршрути ендпоінтів та lifespan
│   ├── schemas.py         # Pydantic схеми запитів та відповідей
│   ├── service.py         # Бізнес-логіка, оркестрація моделей та інференсу
│   ├── data_loader.py     # Завантаження даних та усунення leakage
│   ├── features.py        # ColumnTransformer та агрегація важливості ознак
│   └── models/            # Реалізації RF, LightGBM та Keras MLP
├── streamlit_app/         # Клієнтський інтерфейс
│   ├── app.py             # Головна сторінка та стан системи
│   ├── api_client.py      # HTTP REST клієнт
│   └── pages/             # 1_EDA.py, 2_Model_Training.py, 3_Model_Comparison.py, 4_Forecasting.py
├── notebooks/             # client_demo.ipynb
├── saved_models/          # Збережені артефакти (.joblib, .keras, _meta.json)
├── tests/                 # Тести API, моделей та правил витоку
├── requirements.txt
├── .env.example
└── README.md
```

---

## 2. Встановлення та конфігурація

### Створення та активація віртуального оточення
```powershell
cd c:\Users\userstudent\uni\Проект\repo\lab2\app_sales_forecasting
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Конфігурація оточення (`.env`)
Скопіюйте `.env.example` у `.env`:
```powershell
cp .env.example .env
```
Змінні середовища:
- `DATA_DIR`: шлях до каталогу з CSV-файлами (за замовчуванням `../../data`).
- `API_URL`: адреса бекенд-сервісу (за замовчуванням `http://127.0.0.1:8000`).

---

## 3. Точні команди для запуску

### Запуск FastAPI мікросервісу
Виконується з каталогу `app_sales_forecasting`:
```powershell
cd c:\Users\userstudent\uni\Проект\repo\lab2\app_sales_forecasting
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Документація Swagger UI буде доступна за адресою: `http://127.0.0.1:8000/docs`.

### Запуск Streamlit інтерфейсу
У новому терміналі з каталогу `app_sales_forecasting`:
```powershell
cd c:\Users\userstudent\uni\Проект\repo\lab2\app_sales_forecasting
streamlit run streamlit_app/app.py
```
Веб-інтерфейс відкриється за адресою: `http://localhost:8501`.

### Запуск автоматизованих тестів
```powershell
cd c:\Users\userstudent\uni\Проект\repo\lab2\app_sales_forecasting
pytest tests -v
```

### Запуск клієнтського ноутбука
```powershell
cd c:\Users\userstudent\uni\Проект\repo\lab2\app_sales_forecasting
jupyter lab notebooks/client_demo.ipynb
```

---

## 4. Опис API Ендпоінтів та приклади `curl`

### 1. `GET /health`
Перевірка стану сервісу та доступних навчених моделей.
```bash
curl -X GET http://127.0.0.1:8000/health
```

### 2. `GET /data/info`
Отримання списку магазинів, діапазонів дат і параметрів для налаштування.
```bash
curl -X GET http://127.0.0.1:8000/data/info
```

### 3. `POST /train`
Запуск навчання моделі.
```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "LightGBM",
    "model_name": "LGBM_prod",
    "features": ["Promo", "SchoolHoliday", "logMonthSales", "CompetitionDistance", "Store", "StoreType", "Assortment", "weekday", "month"],
    "hyperparameters": {"n_estimators": 500, "learning_rate": 0.08, "max_depth": 5}
  }'
```

### 4. `GET /models` та `GET /models/{name}`
Отримання списку та деталей збережених моделей.
```bash
curl -X GET http://127.0.0.1:8000/models
curl -X GET http://127.0.0.1:8000/models/LGBM_prod
```

### 5. `POST /predict` (Точковий прогноз)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "LGBM_prod",
    "record": {
      "Store": 1,
      "Date": "2015-05-01",
      "Open": 1,
      "Promo": 1
    }
  }'
```

### 6. `POST /forecast` (Holdout та Сценарний прогноз)
```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "LGBM_prod",
    "stores": [1, 2],
    "start_date": "2015-04-01",
    "end_date": "2015-04-14"
  }'
```

---

## 5. Місце зберігання артефактів
Усі навчені моделі та препроцесори зберігаються у каталозі `app_sales_forecasting/saved_models/`:
- `{model_name}_rf.joblib` або `{model_name}_lgb.joblib` — ваги моделей RF/LightGBM.
- `{model_name}_mlp.keras` та `{model_name}_mlp_scalers.joblib` — архітектура та скейлери Keras MLP.
- `{model_name}_preprocessor.joblib` — `ColumnTransformer` препроцесор ознак.
- `{model_name}_meta.json` — метадані, метрики, криві навчання та важливість ознак.

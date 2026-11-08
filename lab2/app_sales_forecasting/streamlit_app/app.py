"""
Streamlit main entrypoint for the Sales Forecasting Demo Pipeline.
"""

import streamlit as st
import pandas as pd
from api_client import SalesForecastingAPIClient

st.set_page_config(
    page_title="Rossmann Sales Forecasting | Demo Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

client = SalesForecastingAPIClient()

st.title("📈 Rossmann Store Sales Forecasting Demo Pipeline")
st.markdown("""
Ця система демонструє повноцінний ML-конвеєр для щоденного прогнозування виторгу мережі з **100 магазинів Rossmann**.
Архітектура проєкту побудована за принципом мікросервісів:
- **FastAPI ML Microservice**: ізольований бекенд для навчання моделей (**Random Forest**, **LightGBM**, **Keras MLP**), валідації та інференсу.
- **Streamlit Frontend**: інтерактивний клієнтський веб-інтерфейс (EDA, запуск навчання, порівняння метрик, сценарний прогноз).
- **Notebook Client**: аналітичний REST-клієнт для автоматизованого виконання задач.
""")

# Backend Status Card
health = client.check_health()
col1, col2, col3 = st.columns([1.5, 2, 1.5])

with col1:
    if health.get("status") == "ok":
        st.success("🟢 **FastAPI Service:** Online (127.0.0.1:8000)")
    else:
        st.error(f"🔴 **FastAPI Service:** Unreachable ({health.get('error', 'Check if Uvicorn is running')})")

with col2:
    avail_models = health.get("available_models", [])
    st.info(f"🧠 **Завантажені навчені моделі:** {len(avail_models)} ({', '.join(avail_models) if avail_models else 'Немає'})")

with col3:
    if st.button("🔄 Оновити стан сервісу"):
        st.rerun()

st.divider()

st.subheader("📌 Навігація по розділах застосунку")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    ### 📊 1. Дослідження даних (EDA)
    - Статистичний огляд 100 магазинів
    - Розподіл Sales та logSales
    - Часові ряди (Daily/Weekly/Monthly)
    - Вплив акцій (Promo) та свят
    - Аналіз зв'язку Sales з Customers
    """)
    st.page_link("pages/1_EDA.py", label="Перейти до EDA", icon="🔍")

with c2:
    st.markdown("""
    ### ⚙️ 2. Навчання моделей
    - Вибір алгоритму: RF, LightGBM, MLP
    - Налаштування гіперпараметрів
    - Метрики train / val / test vs Baseline
    - Важливість ознак (Feature Importance)
    - Криві навчання (Learning Curves)
    """)
    st.page_link("pages/2_Model_Training.py", label="Перейти до Навчання", icon="🚀")

with c3:
    st.markdown("""
    ### ⚖️ 3. Порівняння моделей
    - Зведена таблиця метрик усіх моделей
    - Діаграми порівняння RMSE, MAE, WAPE
    - Накладання фактичного та прогнозного ряду на валідації
    """)
    st.page_link("pages/3_Model_Comparison.py", label="Перейти до Порівняння", icon="📈")

with c4:
    st.markdown("""
    ### 🔮 4. Прогнозування (Forecasting)
    - **Holdout Forecast**: ретроспективний тест на періоді 2015 року
    - **Scenario Forecast**: симуляція майбутнього виторгу
    - Ефект увімкнення/вимкнення Promo
    - Експорт результатів у CSV
    """)
    st.page_link("pages/4_Forecasting.py", label="Перейти до Прогнозу", icon="🔮")

st.divider()
st.caption("Курс: «Проєкт з розробки системи аналізу даних» | Керівник: проф. Павлишенко Богдан | Виконав: студент групи ФЕІ-43 Гріджак Олександр, Львів 2026")


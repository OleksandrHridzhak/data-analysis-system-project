"""
Page 3: Model Comparison.
Presents validation and test metrics across all saved models, bar charts, and overlaid actual vs predicted time series for a selected store.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from api_client import SalesForecastingAPIClient

st.set_page_config(page_title="Model Comparison | Rossmann Sales", page_icon="⚖️", layout="wide")

st.title("⚖️ Порівняння навчених моделей")
st.markdown("Зіставлення точності та часових прогнозів для всіх збережених моделей на валідаційному та тестовому сплітах.")

client = SalesForecastingAPIClient()

health = client.check_health()
if health.get("status") != "ok":
    st.error("⚠️ FastAPI мікросервіс недоступний! Переконайтесь, що бекенд запущено на `http://127.0.0.1:8000`.")
    st.stop()

# Fetch all models
try:
    models_list = client.list_models()
except Exception as e:
    st.error(f"Помилка отримання моделей: {e}")
    st.stop()

if not models_list:
    st.warning("⚠️ Немає жодної збереженої моделі. Спочатку перейдіть на сторінку **2. Model Training** та натренуйте моделі!")
    st.page_link("pages/2_Model_Training.py", label="Перейти до тренування моделей", icon="🚀")
    st.stop()

st.sidebar.header("Параметри порівняння")
model_names = [m["model_name"] for m in models_list]
selected_models = st.sidebar.multiselect("Обрані моделі для порівняння:", options=model_names, default=model_names)

metric_to_plot = st.sidebar.selectbox("Метрика для графіків:", ["rmse_Sales", "mae_Sales", "wape_Sales", "r2_Sales", "rmse_logSales"])
target_split = st.sidebar.radio("Спліт для порівняння:", ["test", "validation", "train"], horizontal=True)

# Build comparison DataFrame
rows = []
for m in models_list:
    if m["model_name"] in selected_models:
        metrics = m.get("metrics", {})
        split_m = metrics.get(target_split, {})
        rows.append({
            "Модель": m["model_name"],
            "Тип": m["model_type"],
            "Час навч. (с)": m.get("training_time_sec", 0.0),
            "К-сть ознак": len(m.get("features", [])),
            "RMSE (Sales)": split_m.get("rmse_Sales"),
            "MAE (Sales)": split_m.get("mae_Sales"),
            "WAPE (%)": f"{split_m.get('wape_Sales', 0) * 100:.2f}%",
            "R²": split_m.get("r2_Sales"),
            "RMSE (logSales)": split_m.get("rmse_logSales"),
            "raw_metric": split_m.get(metric_to_plot)
        })

comp_df = pd.DataFrame(rows)

st.subheader(f"1. Зведена таблиця метрик на спліті: {target_split.upper()}")
st.dataframe(comp_df.drop(columns=["raw_metric"]), use_container_width=True)

# Bar chart of selected metric
st.subheader(f"2. Діаграма порівняння: {metric_to_plot} ({target_split.upper()})")
if not comp_df.empty:
    fig_bar = px.bar(
        comp_df, x="Модель", y="raw_metric", color="Тип",
        title=f"Порівняння моделей за метрикою {metric_to_plot}",
        text="raw_metric",
        labels={"raw_metric": metric_to_plot}
    )
    fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# Overlaid time series for selected store
st.subheader("3. Накладання фактичного та прогнозного рядів (Overlaid Time Series)")
c_st, c_range = st.columns([1, 2])

data_info = client.get_data_info()
store_ids = data_info.get("store_ids", [1])

with c_st:
    sel_store = st.selectbox("Оберіть номер магазину (Store ID):", store_ids, index=0)

with c_range:
    test_range = data_info.get("test_date_range", {"start": "2015-04-01", "end": "2015-07-31"})
    start_date = st.date_input("Початкова дата", pd.to_datetime(test_range["start"]).date())
    end_date = st.date_input("Кінцева дата", pd.to_datetime(test_range["start"]).date() + pd.Timedelta(days=45))

if st.button("📊 Відобразити часовий графік прогнозу"):
    with st.spinner("Запит щоденних прогнозів через API..."):
        fig_overlay = go.Figure()
        actual_plotted = False

        for m_name in selected_models:
            try:
                fc_res = client.forecast(
                    model_name=m_name,
                    stores=[int(sel_store)],
                    start_date=str(start_date),
                    end_date=str(end_date)
                )
                items = fc_res.get("forecasts", [])
                df_store_fc = pd.DataFrame(items)

                if not actual_plotted and "actual_sales" in df_store_fc.columns and df_store_fc["actual_sales"].notnull().any():
                    fig_overlay.add_trace(go.Scatter(
                        x=df_store_fc["Date"], y=df_store_fc["actual_sales"],
                        mode="lines+markers", name="Фактичні продажі (Actual)",
                        line=dict(color="black", width=2.5)
                    ))
                    actual_plotted = True

                fig_overlay.add_trace(go.Scatter(
                    x=df_store_fc["Date"], y=df_store_fc["predicted_sales"],
                    mode="lines", name=f"Прогноз: {m_name}",
                    line=dict(width=2)
                ))
            except Exception as e:
                st.warning(f"Не вдалося побудувати прогноз для {m_name}: {e}")

        fig_overlay.update_layout(
            title=f"Фактичний vs Прогнозні продажі для Магазину №{sel_store} ({start_date} – {end_date})",
            xaxis_title="Дата",
            yaxis_title="Виторг ($)",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_overlay, use_container_width=True)


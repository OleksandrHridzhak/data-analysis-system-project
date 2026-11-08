"""
Page 4: Sales Forecasting (Holdout & Scenario Simulation).
Strictly accesses backend via REST calls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from api_client import SalesForecastingAPIClient

st.set_page_config(page_title="Forecasting | Rossmann Sales", page_icon="🔮", layout="wide")

st.title("🔮 Модуль прогнозування та сценарного аналізу")
st.markdown("Ретроспективний аналіз точності на відкладеній вибірці (Holdout) та інтерактивна симуляція майбутнього виторгу.")

client = SalesForecastingAPIClient()

health = client.check_health()
if health.get("status") != "ok":
    st.error("⚠️ FastAPI мікросервіс недоступний! Переконайтесь, що бекенд запущено на `http://127.0.0.1:8000`.")
    st.stop()

# Check available models
models_list = client.list_models()
if not models_list:
    st.warning("⚠️ Немає жодної збереженої моделі. Для здійснення прогнозу спочатку натренуйте модель на сторінці **2. Model Training**!")
    st.page_link("pages/2_Model_Training.py", label="Перейти до сторінки тренування", icon="🚀")
    st.stop()

data_info = client.get_data_info()
model_names = [m["model_name"] for m in models_list]

tab_holdout, tab_scenario = st.tabs(["📅 Holdout Forecast (2015)", "🔮 Scenario Simulation (Майбутні сценарії)"])

# ----------------- TAB 1: HOLDOUT FORECAST -----------------
with tab_holdout:
    st.subheader("1. Ретроспективний прогноз на даних 2015 року")
    st.markdown("Оберіть натреновану модель, один або кілька магазинів та діапазон дат у 2015 році для зіставлення з фактом.")

    c_m, c_st, c_dt = st.columns([1, 1.5, 1.5])
    with c_m:
        h_model = st.selectbox("Модель для прогнозу:", model_names, key="h_model")
    with c_st:
        all_stores = data_info.get("store_ids", [1])
        h_stores = st.multiselect("Магазини:", options=all_stores, default=[1, 2], key="h_stores")
    with c_dt:
        min_2015 = pd.to_datetime("2015-01-01").date()
        max_2015 = pd.to_datetime("2015-07-31").date()
        h_range = st.date_input("Діапазон дат (2015):", [min_2015, min_2015 + pd.Timedelta(days=60)], min_value=min_2015, max_value=max_2015)

    if st.button("📈 Побудувати Holdout Прогноз", type="primary"):
        if not h_stores:
            st.error("Оберіть хоча б один магазин!")
        elif len(h_range) != 2:
            st.error("Оберіть початок і кінець діапазону!")
        else:
            with st.spinner("Отримання прогнозу та розрахунок метрик..."):
                try:
                    res = client.forecast(
                        model_name=h_model,
                        stores=h_stores,
                        start_date=str(h_range[0]),
                        end_date=str(h_range[1])
                    )
                    items = res.get("forecasts", [])
                    df_fc = pd.DataFrame(items)

                    if not df_fc.empty:
                        # Aggregate metrics card
                        agg_m = res.get("aggregate_metrics")
                        if agg_m:
                            st.markdown("#### Загальні агреговані метрики за обраними магазинами:")
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("RMSE (Sales)", f"${agg_m['rmse_Sales']:,.2f}")
                            m2.metric("MAE (Sales)", f"${agg_m['mae_Sales']:,.2f}")
                            m3.metric("WAPE", f"{agg_m['wape_Sales']*100:.2f}%")
                            m4.metric("R²", agg_m["r2_Sales"])
                            m5.metric("RMSE (logSales)", agg_m["rmse_logSales"])

                        # Per-store metrics table
                        st_m = res.get("per_store_metrics")
                        if st_m:
                            st.markdown("#### Метрики за окремими магазинами:")
                            st_df = pd.DataFrame.from_dict(st_m, orient="index").reset_index().rename(columns={"index": "Store"})
                            st.dataframe(st_df, use_container_width=True)

                        # Plot Actual vs Forecast Time Series
                        fig_h = go.Figure()
                        for s_id in h_stores:
                            s_data = df_fc[df_fc["Store"] == s_id]
                            fig_h.add_trace(go.Scatter(
                                x=s_data["Date"], y=s_data["actual_sales"],
                                mode="lines", name=f"Магазин {s_id} (Факт)",
                                line=dict(dash="dot")
                            ))
                            fig_h.add_trace(go.Scatter(
                                x=s_data["Date"], y=s_data["predicted_sales"],
                                mode="lines+markers", name=f"Магазин {s_id} (Прогноз)"
                            ))

                        fig_h.update_layout(
                            title=f"Фактичний та прогнозний щоденний виторг ({h_range[0]} – {h_range[1]})",
                            xaxis_title="Дата",
                            yaxis_title="Виторг ($)",
                            template="plotly_white",
                            margin=dict(l=30, r=20, t=50, b=30),
                            autosize=True
                        )
                        st.plotly_chart(fig_h, use_container_width=True)

                        # CSV Download button
                        csv_data = df_fc.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Завантажити результати прогнозу (CSV)",
                            data=csv_data,
                            file_name=f"holdout_forecast_{h_model}_{h_range[0]}_{h_range[1]}.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Помилка прогнозу: {e}")

# ----------------- TAB 2: SCENARIO FORECAST -----------------
with tab_scenario:
    st.subheader("2. Сценарне моделювання та оцінка впливу промо-акцій (Promo On/Off)")
    st.markdown("Задайте параметри майбутнього періоду: статус роботи (Open), акції (Promo), свята, та оцініть зміну очікуваного виторгу.")

    c_sm, c_sst, c_sdt = st.columns([1, 1, 1.5])
    with c_sm:
        s_model = st.selectbox("Модель:", model_names, key="s_model")
    with c_sst:
        s_store = st.selectbox("Магазин (Store):", data_info.get("store_ids", [1]), key="s_store")
    with c_sdt:
        future_start = pd.to_datetime("2015-08-01").date()
        future_range = st.date_input("Майбутній період:", [future_start, future_start + pd.Timedelta(days=14)], min_value=future_start)

    st.markdown("#### Базові налаштування діапазону:")
    c_p, c_sch, c_sh, c_op = st.columns(4)
    with c_p:
        def_promo = st.radio("Promo для періоду:", [1, 0], format_func=lambda x: "Увімкнено (1)" if x == 1 else "Вимкнено (0)")
    with c_sch:
        def_school = st.selectbox("SchoolHoliday:", [0, 1])
    with c_sh:
        def_state = st.selectbox("StateHoliday:", ["0", "a", "b", "c"])
    with c_op:
        def_open = st.selectbox("Open (1=відкрито, 0=закрито):", [1, 0])

    # Per-day override table
    if len(future_range) == 2:
        date_list = pd.date_range(future_range[0], future_range[1], freq="D").strftime("%Y-%m-%d").tolist()
        scenario_init = []
        for d_str in date_list:
            dt = pd.to_datetime(d_str)
            is_sunday = (dt.weekday() == 6)
            is_op = 0 if is_sunday else def_open
            scenario_init.append({
                "Date": d_str,
                "Store": int(s_store),
                "Open": is_op,
                "Promo": def_promo,
                "SchoolHoliday": def_school,
                "StateHoliday": def_state
            })
        st.markdown("#### Таблиця подекадних/щоденних коригувань (можна редагувати значення клітинок):")
        edited_scenario = st.data_editor(pd.DataFrame(scenario_init), use_container_width=True, num_rows="fixed")

        if st.button("🚀 Розрахувати сценарний прогноз та Promo Effect", type="primary"):
            with st.spinner("Симуляція сценарію та аналіз дельти виторгу..."):
                try:
                    overrides = edited_scenario.to_dict(orient="records")

                    # Scenario 1: User scenario
                    res_user = client.forecast(
                        model_name=s_model,
                        stores=[int(s_store)],
                        start_date=str(future_range[0]),
                        end_date=str(future_range[1]),
                        scenario_overrides=overrides
                    )

                    # Scenario 2: Promo inverted for effect comparison
                    inverted_overrides = []
                    for row in overrides:
                        r_inv = dict(row)
                        r_inv["Promo"] = 0 if row["Promo"] == 1 else 1
                        inverted_overrides.append(r_inv)

                    res_inv = client.forecast(
                        model_name=s_model,
                        stores=[int(s_store)],
                        start_date=str(future_range[0]),
                        end_date=str(future_range[1]),
                        scenario_overrides=inverted_overrides
                    )

                    df_user = pd.DataFrame(res_user.get("forecasts", []))
                    df_inv = pd.DataFrame(res_inv.get("forecasts", []))

                    total_user_sales = df_user["predicted_sales"].sum()
                    total_inv_sales = df_inv["predicted_sales"].sum()
                    diff = total_user_sales - total_inv_sales

                    st.markdown("### Порівняння ефекту Promo (Увімкнено проти Вимкнено)")
                    col_tot1, col_tot2, col_diff = st.columns(3)
                    col_tot1.metric("Сумарний виторг (Заданий сценарій)", f"${total_user_sales:,.0f}")
                    col_tot2.metric("Сумарний виторг (Інвертований Promo)", f"${total_inv_sales:,.0f}")
                    col_diff.metric("Різниця виторгу (Promo Lift)", f"${diff:+,.0f}", delta=f"{diff/max(1, total_inv_sales)*100:+.1f}%")

                    fig_scen = go.Figure()
                    fig_scen.add_trace(go.Bar(
                        x=df_user["Date"], y=df_user["predicted_sales"],
                        name="Поточний сценарій", marker_color="#1f77b4"
                    ))
                    fig_scen.add_trace(go.Bar(
                        x=df_inv["Date"], y=df_inv["predicted_sales"],
                        name="Інвертований Promo", marker_color="#ff7f0e", opacity=0.7
                    ))
                    fig_scen.update_layout(
                        barmode="group",
                        title=f"Прогнозовані продажі: Сценарій vs Альтернативний Promo (Магазин №{s_store})",
                        xaxis_title="Дата",
                        yaxis_title="Очікуваний щоденний виторг ($)",
                        template="plotly_white",
                        margin=dict(l=30, r=20, t=50, b=30),
                        autosize=True
                    )
                    st.plotly_chart(fig_scen, use_container_width=True)

                except Exception as e:
                    st.error(f"Помилка моделювання сценарію: {e}")


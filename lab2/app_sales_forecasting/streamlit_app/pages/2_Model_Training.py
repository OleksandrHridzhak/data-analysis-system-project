"""
Page 2: Model Training via FastAPI REST API.
Allows user to select model architecture, features, hyperparameters, and trigger training.
Displays metrics comparison with Baseline, scatter plots, residual distribution, feature importances, and learning curves.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from api_client import SalesForecastingAPIClient

st.set_page_config(page_title="Model Training | Rossmann Sales", page_icon="⚙️", layout="wide")

st.title("⚙️ Навчання та оцінювання моделей")
st.markdown("Запуск тренування моделей машинного навчання через виклики до **FastAPI REST API**.")

client = SalesForecastingAPIClient()

# Check service connection
health = client.check_health()
if health.get("status") != "ok":
    st.error("⚠️ FastAPI мікросервіс недоступний! Переконайтесь, що бекенд запущено на `http://127.0.0.1:8000`.")
    st.stop()

# Load data info from API
try:
    data_info = client.get_data_info()
except Exception as e:
    st.error(f"Помилка отримання метаданих через API: {e}")
    st.stop()

st.sidebar.header("Параметри моделювання")

# 1. Model Selection
model_type = st.sidebar.selectbox(
    "Тип моделі",
    ["RandomForest", "LightGBM", "KerasMLP"],
    help="RandomForestRegressor, LGBMRegressor або Keras Deep MLP"
)

default_name = f"{model_type}_v1"
model_name = st.sidebar.text_input("Назва моделі", value=default_name)

# 2. Features selection
all_features = data_info.get("all_candidate_features", [])
selected_features = st.sidebar.multiselect(
    "Ознаки для навчання (Features)",
    options=all_features,
    default=all_features,
    help="Customers виключено з вибірки через витік цілі (target leakage)"
)

if not selected_features:
    st.sidebar.warning("Виберіть хоча б одну ознаку!")

# 3. Hyperparameters
st.sidebar.subheader("Гіперпараметри")
hyperparams = {}

if model_type == "RandomForest":
    n_est = st.sidebar.slider("n_estimators", min_value=10, max_value=200, value=100, step=10)
    max_d = st.sidebar.selectbox("max_depth", [None, 10, 15, 20, 25])
    min_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=20, value=1)
    hyperparams = {"n_estimators": n_est, "max_depth": max_d, "min_samples_leaf": min_leaf}

elif model_type == "LightGBM":
    n_est = st.sidebar.slider("n_estimators (max)", min_value=100, max_value=1500, value=1000, step=100)
    lr = st.sidebar.slider("learning_rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
    n_leaves = st.sidebar.slider("num_leaves", min_value=20, max_value=150, value=100, step=10)
    max_d = st.sidebar.slider("max_depth", min_value=3, max_value=15, value=5)
    cols = st.sidebar.slider("colsample_bytree", min_value=0.3, max_value=1.0, value=0.7, step=0.1)
    hyperparams = {
        "n_estimators": n_est, "learning_rate": lr, "num_leaves": n_leaves,
        "max_depth": max_d, "colsample_bytree": cols
    }

elif model_type == "KerasMLP":
    hidden = st.sidebar.selectbox("hidden_layer_sizes", ["[512, 256]", "[256, 128]", "[512, 256, 128]"])
    hidden_list = eval(hidden)
    dropout = st.sidebar.slider("dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
    lr = st.sidebar.select_slider("learning_rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-4)
    epochs = st.sidebar.slider("epochs", min_value=5, max_value=50, value=20, step=5)
    batch_size = st.sidebar.select_slider("batch_size", options=[128, 256, 512, 1024], value=512)
    hyperparams = {
        "hidden_layer_sizes": hidden_list, "dropout": dropout,
        "learning_rate": lr, "epochs": epochs, "batch_size": batch_size
    }

# Main Area: Train button
col_btn, col_info = st.columns([1, 3])
with col_btn:
    train_clicked = st.button("🚀 Train Model", type="primary", use_container_width=True)

with col_info:
    st.info(f"Обрано алгоритм: **{model_type}** | Кількість ознак: **{len(selected_features)}** | Модель зберігається на сервері під іменем `{model_name}`.")

if train_clicked:
    if not selected_features:
        st.error("Помилка: не обрано жодної ознаки для навчання.")
        st.stop()

    with st.spinner(f"Триває тренування моделі '{model_name}' на стороні FastAPI мікросервісу..."):
        try:
            train_result = client.train_model(
                model_type=model_type,
                model_name=model_name,
                features=selected_features,
                hyperparameters=hyperparams
            )
            st.session_state["latest_training"] = train_result
            st.success(f"✅ Модель '{model_name}' успішно натренована за {train_result['training_time_sec']} сек!")
        except Exception as e:
            st.error(f"Помилка під час виклику POST /train: {e}")
            st.stop()

# Display results if available in session_state or fetched
res = st.session_state.get("latest_training")
if res:
    st.divider()
    st.subheader(f"📊 Результати тренування: {res['model_name']} ({res['model_type']})")

    # Metrics table vs Baseline
    m = res["metrics"]
    b = res["baseline_metrics"]

    metrics_df = pd.DataFrame({
        "Спліт": ["Train (2013-2014)", "Validation (< 2015-04)", "Test (>= 2015-04)",
                  "Train Baseline", "Val Baseline", "Test Baseline"],
        "RMSE (logSales)": [
            m["train"]["rmse_logSales"], m["validation"]["rmse_logSales"], m["test"]["rmse_logSales"],
            b["train"]["rmse_logSales"], b["validation"]["rmse_logSales"], b["test"]["rmse_logSales"]
        ],
        "MAE (Sales, $)": [
            f"${m['train']['mae_Sales']:,.2f}", f"${m['validation']['mae_Sales']:,.2f}", f"${m['test']['mae_Sales']:,.2f}",
            f"${b['train']['mae_Sales']:,.2f}", f"${b['validation']['mae_Sales']:,.2f}", f"${b['test']['mae_Sales']:,.2f}"
        ],
        "RMSE (Sales, $)": [
            f"${m['train']['rmse_Sales']:,.2f}", f"${m['validation']['rmse_Sales']:,.2f}", f"${m['test']['rmse_Sales']:,.2f}",
            f"${b['train']['rmse_Sales']:,.2f}", f"${b['validation']['rmse_Sales']:,.2f}", f"${b['test']['rmse_Sales']:,.2f}"
        ],
        "WAPE (%)": [
            f"{m['train']['wape_Sales']*100:.2f}%", f"{m['validation']['wape_Sales']*100:.2f}%", f"{m['test']['wape_Sales']*100:.2f}%",
            f"{b['train']['wape_Sales']*100:.2f}%", f"{b['validation']['wape_Sales']*100:.2f}%", f"{b['test']['wape_Sales']*100:.2f}%"
        ],
        "R²": [
            m["train"]["r2_Sales"], m["validation"]["r2_Sales"], m["test"]["r2_Sales"],
            b["train"]["r2_Sales"], b["validation"]["r2_Sales"], b["test"]["r2_Sales"]
        ]
    })

    st.markdown("### Таблиця метрик моделі проти Naive Baseline")
    st.dataframe(metrics_df, use_container_width=True)

    # Visualizations: Feature Importances & Learning Curve
    col_fi, col_lc = st.columns(2)

    fi_data = res.get("feature_importances")
    with col_fi:
        if fi_data:
            fi_df = pd.DataFrame(list(fi_data.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                title=f"Важливість вихідних ознак ({res['model_type']})",
                color="Importance", color_continuous_scale="Purples"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance доступна для моделей Random Forest та LightGBM.")

    lc_data = res.get("learning_curve")
    with col_lc:
        if lc_data and "train_loss" in lc_data and lc_data["train_loss"]:
            t_loss = lc_data["train_loss"]
            v_loss = lc_data.get("val_loss", [])
            epochs_x = list(range(1, len(t_loss) + 1))

            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=epochs_x, y=t_loss, mode="lines", name="Train Loss", line=dict(color="#1f77b4", width=2)))
            if v_loss and len(v_loss) == len(t_loss):
                fig_lc.add_trace(go.Scatter(x=epochs_x, y=v_loss, mode="lines", name="Val Loss", line=dict(color="#d62728", width=2, dash="dash")))

            fig_lc.update_layout(
                title=f"Крива навчання (Learning Curve) - {res['model_type']}",
                xaxis_title="Ітерація / Епоха",
                yaxis_title="RMSE / MAE Loss",
                template="plotly_white"
            )
            st.plotly_chart(fig_lc, use_container_width=True)
        else:
            st.info("Криві навчання доступні для LightGBM (iterative boosting) та Keras MLP (per-epoch history).")

    # Diagnostics: Run test holdout sample to plot Scatter & Residuals
    st.markdown("### Діагностичні графіки (Actual vs Predicted та Residual Distribution)")
    with st.spinner("Завантаження прогнозу на тестовому спліті для діагностики..."):
        try:
            fc_sample = client.forecast(
                model_name=res["model_name"],
                stores=[1, 2, 3],
                start_date="2015-04-01",
                end_date="2015-05-15"
            )
            fc_items = fc_sample.get("forecasts", [])
            df_diag = pd.DataFrame(fc_items)
            df_diag = df_diag[(df_diag["is_open"] == 1) & (df_diag["actual_sales"].notnull())]

            c_scat, c_resid = st.columns(2)
            with c_scat:
                fig_scat = px.scatter(
                    df_diag, x="actual_sales", y="predicted_sales",
                    title="Actual vs Predicted Sales (Test Sample)",
                    labels={"actual_sales": "Фактичний виторг ($)", "predicted_sales": "Прогнозний виторг ($)"},
                    opacity=0.7, color="Store"
                )
                max_val = max(df_diag["actual_sales"].max(), df_diag["predicted_sales"].max())
                fig_scat.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Ідеальна лінія", line=dict(color="red", dash="dash")))
                st.plotly_chart(fig_scat, use_container_width=True)

            with c_resid:
                df_diag["residual"] = df_diag["actual_sales"] - df_diag["predicted_sales"]
                fig_res = px.histogram(
                    df_diag, x="residual", nbins=30,
                    title="Розподіл залишків (Residual Distribution: Actual - Predicted)",
                    color_discrete_sequence=["#ff7f0e"],
                    marginal="rug"
                )
                st.plotly_chart(fig_res, use_container_width=True)
        except Exception as err:
            st.warning(f"Діагностичні графіки пропущено: {err}")


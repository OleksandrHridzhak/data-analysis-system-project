"""
Page 1: Exploratory Data Analysis (EDA) following nb08.
All charts built using Plotly.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EDA | Rossmann Sales", page_icon="📊", layout="wide")

st.title("📊 Дослідницький аналіз даних (EDA)")
st.markdown("Аналіз набору даних 100 магазинів мережі Rossmann згідно з методикою лабораторної роботи.")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


@st.cache_data
def load_eda_data():
    train_path = os.path.join(DATA_DIR, "rossm_train_df.csv")
    test_path = os.path.join(DATA_DIR, "rossm_test_df.csv")
    tr = pd.read_csv(train_path, dtype={"StateHoliday": str})
    te = pd.read_csv(test_path, dtype={"StateHoliday": str})
    df = pd.concat([tr, te], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)
    return df


df = load_eda_data()

# Sidebar filters applied to all charts
st.sidebar.header("🔍 Фільтри даних")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input("Діапазон дат", [min_date, max_date], min_value=min_date, max_value=max_date)
all_stores = sorted(df["Store"].unique().tolist())
selected_stores = st.sidebar.multiselect("Магазини (пусто = всі 100)", options=all_stores, default=[])

only_open = st.sidebar.checkbox("Лише відкриті дні (Open == 1)", value=True)

# Apply filters
filtered_df = df.copy()
if len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df["Date"] >= start_d) & (filtered_df["Date"] <= end_d)]

if selected_stores:
    filtered_df = filtered_df[filtered_df["Store"].isin(selected_stores)]

if only_open:
    filtered_df = filtered_df[filtered_df["Open"] == 1]

# Section 1: Overview
st.subheader("1. Загальний огляд набору даних")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Кількість записів", f"{len(filtered_df):,}")
col2.metric("Кількість магазинів", filtered_df["Store"].nunique())
col3.metric("Початок вибірки", str(filtered_df["Date"].min().strftime('%Y-%m-%d')) if not filtered_df.empty else "-")
col4.metric("Кінець вибірки", str(filtered_df["Date"].max().strftime('%Y-%m-%d')) if not filtered_df.empty else "-")
col5.metric("Середній щоденний виторг", f"${filtered_df['Sales'].mean():,.0f}" if not filtered_df.empty else "-")

with st.expander("📋 Зразок даних та типи колонок", expanded=False):
    st.write(filtered_df.head(10))
    st.write("Типи даних:")
    st.write(pd.DataFrame(filtered_df.dtypes, columns=["Dtype"]).T)

st.divider()

# Section 2: Distributions of Sales & logSales
st.subheader("2. Розподіли виторгу: Sales та logSales")
c1, c2 = st.columns(2)

with c1:
    fig_sales = px.histogram(
        filtered_df, x="Sales", nbins=50,
        title="Гістограма розподілу щоденного виторгу (Sales)",
        color_discrete_sequence=["#1f77b4"],
        marginal="box"
    )
    fig_sales.update_layout(bargap=0.1)
    st.plotly_chart(fig_sales, use_container_width=True)

with c2:
    fig_log = px.histogram(
        filtered_df, x="logSales", nbins=50,
        title="Гістограма логарифмованого виторгу (logSales)",
        color_discrete_sequence=["#2ca02c"],
        marginal="box"
    )
    fig_log.update_layout(bargap=0.1)
    st.plotly_chart(fig_log, use_container_width=True)

st.divider()

# Section 3: Time series and Aggregations
st.subheader("3. Динаміка продажів у часі (Агрегації: день / тиждень / місяць)")
agg_choice = st.radio("Рівень агрегації часового ряду:", ["День", "Тиждень", "Місяць"], horizontal=True)

freq_map = {"День": "D", "Тиждень": "W-MON", "Місяць": "MS"}
ts_data = filtered_df.set_index("Date").resample(freq_map[agg_choice])["Sales"].sum().reset_index()

fig_ts = px.line(
    ts_data, x="Date", y="Sales",
    title=f"Сумарний виторг за часом ({agg_choice.lower()})",
    labels={"Sales": "Сумарні продажі ($)", "Date": "Дата"}
)
fig_ts.update_traces(line_color="#ff7f0e", line_width=2)
st.plotly_chart(fig_ts, use_container_width=True)

c_top, c_stores = st.columns([1, 2])
with c_top:
    top_stores = filtered_df.groupby("Store")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(10)
    fig_top = px.bar(
        top_stores, x="Sales", y="Store", orientation="h",
        title="Топ-10 магазинів за загальним виторгом",
        color="Sales", color_continuous_scale="Viridis"
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)

with c_stores:
    sample_stores = selected_stores if selected_stores else [1, 2, 3, 4, 5]
    store_ts = filtered_df[filtered_df["Store"].isin(sample_stores)]
    fig_store_ts = px.line(
        store_ts, x="Date", y="Sales", color="Store",
        title=f"Щоденний виторг для магазинів: {sample_stores}"
    )
    st.plotly_chart(fig_store_ts, use_container_width=True)

st.divider()

# Section 4: Boxplot by Weekday split by Promo
st.subheader("4. Продажі за днями тижня та вплив промо-акцій (Promo)")
c_box, c_cat = st.columns(2)

with c_box:
    weekday_labels = {1: "Пн", 2: "Вт", 3: "Ср", 4: "Чт", 5: "Пт", 6: "Сб", 7: "Нд"}
    filtered_df["WeekdayName"] = filtered_df["weekday"].map(weekday_labels)
    filtered_df["PromoLabel"] = filtered_df["Promo"].map({0: "Без Promo", 1: "З Promo"})
    
    fig_box = px.box(
        filtered_df, x="WeekdayName", y="Sales", color="PromoLabel",
        category_orders={"WeekdayName": ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]},
        title="Boxplot виторгу за днями тижня (розбиття за Promo)",
        color_discrete_map={"Без Promo": "#1f77b4", "З Promo": "#e377c2"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

with c_cat:
    cat_feature = st.selectbox("Категоріальна ознака для групування:", ["StoreType", "Assortment", "StateHoliday", "SchoolHoliday", "month"])
    cat_sales = filtered_df.groupby(cat_feature)["Sales"].mean().reset_index()
    fig_cat_bar = px.bar(
        cat_sales, x=cat_feature, y="Sales",
        title=f"Середній виторг залежно від '{cat_feature}'",
        color="Sales", color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_cat_bar, use_container_width=True)

st.divider()

# Section 5: Sales vs Customers & CompetitionDistance
st.subheader("5. Взаємозв'язок виторгу з Customers та CompetitionDistance")
c_cust, c_comp = st.columns(2)

with c_cust:
    fig_cust = px.scatter(
        filtered_df.sample(min(len(filtered_df), 3000), random_state=42),
        x="Customers", y="Sales", color="PromoLabel",
        title="Sales vs Customers (Витік цільової змінної для EDA)",
        trendline="ols",
        opacity=0.6
    )
    st.plotly_chart(fig_cust, use_container_width=True)

with c_comp:
    fig_comp = px.scatter(
        filtered_df.sample(min(len(filtered_df), 3000), random_state=42),
        x="CompetitionDistance", y="Sales", color="StoreType",
        title="Sales vs CompetitionDistance (Відстань до конкурента)",
        log_x=True, opacity=0.6
    )
    st.plotly_chart(fig_comp, use_container_width=True)


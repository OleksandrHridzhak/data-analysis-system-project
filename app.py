import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import DATA_FILE
from src.data_generator import generate_real_estate_data
from src.ui.training_view import render_training_view
from src.ui.prediction_view import render_prediction_view

# Streamlit Page Config
st.set_page_config(
    page_title="Real Estate Price Prediction PoC",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """
    Loads dataset from disk. If not generated yet, generates it on the fly.
    """
    if not DATA_FILE.exists():
        df = generate_real_estate_data(n_samples=20000, random_seed=42)
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_FILE, index=False)
        return df
    return pd.read_csv(DATA_FILE)


def main():
    # Sidebar Navigation & Overview
    st.sidebar.title("🏡 Real Estate PoC")
    st.sidebar.markdown("**RandomForest Regression Platform**")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_dataset()
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Navigation")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["🛠️ Training Mode", "🎯 Prediction & Analytics Mode"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.write(f"- Total Properties: **{len(df):,}**")
    st.sidebar.write(f"- Features: **{df.shape[1] - 1}**")
    st.sidebar.write(f"- Median Price: **${df['price'].median():,.0f}**")
    st.sidebar.write(f"- Cities: **{df['city'].nunique()}**")
    
    with st.sidebar.expander("ℹ️ Architectural Boundaries"):
        st.markdown(
            """
            - **Training Feature Selection**: Used only during pipeline fit.
            - **Prediction Inputs**: Individual property values for inference.
            - **Analytics Features**: Slice & group evaluations without retraining.
            - **Training Execution**: Only triggered by explicit button click.
            """
        )

    # Route based on selected mode
    if app_mode == "🛠️ Training Mode":
        render_training_view(df)
    else:
        render_prediction_view(df)


if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
from typing import Optional

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, MODEL_FILE
from src.model_manager import train_and_evaluate, save_model_bundle, load_model_bundle
from src.ui.components import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)


def render_training_view(df: pd.DataFrame):
    st.header("🛠️ Model Training & Hyperparameter Tuning")
    st.markdown(
        """
        Configure features, hyperparameters, and train/test partitions.
        **Training will ONLY execute when you explicitly click the 'Train Model' button.**
        """
    )
    
    col_cfg1, col_cfg2 = st.columns([1, 1], gap="medium")
    
    with col_cfg1:
        st.subheader("1. Feature Selection for Training")
        selected_num = st.multiselect(
            "Numerical Features:",
            options=NUMERICAL_FEATURES,
            default=NUMERICAL_FEATURES,
            help="Select numerical features to include in the training pipeline."
        )
        selected_cat = st.multiselect(
            "Categorical Features:",
            options=CATEGORICAL_FEATURES,
            default=CATEGORICAL_FEATURES,
            help="Select categorical features to one-hot encode in the training pipeline."
        )
        
        st.info(f"Total features selected: **{len(selected_num) + len(selected_cat)}** ({len(selected_num)} numerical, {len(selected_cat)} categorical)")
        
    with col_cfg2:
        st.subheader("2. Hyperparameters & Train/Test Split")
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            n_estimators = st.slider("n_estimators (Trees)", min_value=20, max_value=250, value=80, step=10)
            max_depth = st.slider("max_depth", min_value=4, max_value=25, value=14, step=1)
            min_samples_split = st.slider("min_samples_split", min_value=2, max_value=10, value=4, step=1)
        with col_hp2:
            min_samples_leaf = st.slider("min_samples_leaf", min_value=1, max_value=10, value=2, step=1)
            max_features = st.selectbox("max_features", options=["sqrt", "log2", 0.5, 0.8], index=0)
            test_size = st.slider("Test Set Fraction", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
            
        random_seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    
    # Explicit training button
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        train_clicked = st.button("🚀 Train Model", type="primary", use_container_width=True)
        
    if train_clicked:
        if not (selected_num or selected_cat):
            st.error("Please select at least one feature for training.")
            return
            
        with st.spinner("Training RandomForestRegressor Pipeline... Please wait."):
            try:
                bundle = train_and_evaluate(
                    df=df,
                    numerical_features=selected_num,
                    categorical_features=selected_cat,
                    test_size=test_size,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=int(random_seed)
                )
                
                # Save to disk using joblib
                save_path = save_model_bundle(bundle)
                st.session_state["model_bundle"] = bundle
                st.success(f"Model successfully trained and persisted to `{save_path}`! (Trained at {bundle['trained_at']})")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                return

    # Check if a model exists in session_state or on disk
    bundle = st.session_state.get("model_bundle")
    if bundle is None:
        bundle = load_model_bundle()
        if bundle is not None:
            st.session_state["model_bundle"] = bundle
            st.info(f"Loaded existing trained model from `{MODEL_FILE}` (Trained at {bundle.get('trained_at', 'N/A')}).")
            
    if bundle is not None:
        st.markdown("### 📊 Model Evaluation Performance")
        test_m = bundle["test_metrics"]
        train_m = bundle["train_metrics"]
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("MAE (Test)", f"${test_m['MAE']:,.0f}", delta=f"Train: ${train_m['MAE']:,.0f}", delta_color="inverse")
        m_col2.metric("RMSE (Test)", f"${test_m['RMSE']:,.0f}", delta=f"Train: ${train_m['RMSE']:,.0f}", delta_color="inverse")
        m_col3.metric("R² Score (Test)", f"{test_m['R2']:.4f}", delta=f"Train: {train_m['R2']:.4f}")
        m_col4.metric("MAPE (Test)", f"{test_m['MAPE']:.2f}%", delta=f"Train: {train_m['MAPE']:.2f}%", delta_color="inverse")
        
        st.markdown("---")
        
        # Interactive plots
        st.subheader("📈 Diagnostics & Visualizations")
        tab1, tab2, tab3 = st.tabs(["Actual vs. Predicted", "Residual Analysis", "Feature Importance"])
        
        with tab1:
            fig_avp = plot_actual_vs_predicted(bundle["eval_sample"])
            st.plotly_chart(fig_avp, use_container_width=True)
            
        with tab2:
            fig_res = plot_residuals(bundle["eval_sample"])
            st.plotly_chart(fig_res, use_container_width=True)
            
        with tab3:
            fig_imp = plot_feature_importance(bundle["df_importance"])
            st.plotly_chart(fig_imp, use_container_width=True)
            with st.expander("View Raw Feature Importance Table"):
                st.dataframe(bundle["df_importance"], use_container_width=True)
    else:
        st.warning("No model trained yet. Select features above and click '🚀 Train Model'.")


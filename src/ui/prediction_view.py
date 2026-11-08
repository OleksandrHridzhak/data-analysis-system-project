import io
import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    CITIES_PROFILE,
    PROPERTY_TYPES,
    PROPERTY_CONDITIONS,
    ROOF_TYPES,
    HEATING_TYPES,
    TARGET_COL,
)
from src.model_manager import load_model_bundle
from src.analytics import compute_categorical_aggregation, compute_numerical_sensitivity
from src.ui.components import (
    plot_sensitivity_curve,
    plot_categorical_comparison,
    plot_actual_vs_predicted,
)


def render_prediction_view(df: pd.DataFrame):
    bundle = st.session_state.get("model_bundle")
    if bundle is None:
        bundle = load_model_bundle()
        if bundle is not None:
            st.session_state["model_bundle"] = bundle
            
    if bundle is None:
        st.warning("⚠️ No trained model found! Please go to the **Training** tab first and train a model.")
        return

    pipeline = bundle["pipeline"]
    trained_features = bundle["all_features"]
    trained_num = bundle["numerical_features"]
    trained_cat = bundle["categorical_features"]
    
    st.header("🎯 Prediction & In-Depth Analytics")
    st.caption(f"Active Model trained at: **{bundle.get('trained_at', 'Unknown')}** with **{len(trained_features)}** features.")

    mode = st.radio(
        "Select Operation Mode:",
        options=[
            "🏡 Single Property Prediction & Sensitivity",
            "📁 Batch Predictions & Export",
            "📊 Categorical Aggregations & Analytics",
            "🔍 Analytics Feature Exploration"
        ],
        horizontal=True
    )

    st.markdown("---")

    # =========================================================================
    # 1. SINGLE PROPERTY PREDICTION & SENSITIVITY
    # =========================================================================
    if mode == "🏡 Single Property Prediction & Sensitivity":
        st.subheader("1. Enter Property Characteristics (Prediction Inputs)")
        
        # Split inputs into neat sections
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📍 Location & Type")
            city = st.selectbox("City", options=list(CITIES_PROFILE.keys()), index=0)
            state = CITIES_PROFILE[city]["state"]
            st.text_input("State", value=state, disabled=True)
            prop_type = st.selectbox("Property Type", options=PROPERTY_TYPES, index=0)
            prop_cond = st.selectbox("Property Condition", options=PROPERTY_CONDITIONS, index=2)
            dist_downtown = st.slider("Distance to Downtown (miles)", 0.2, 35.0, 6.5, 0.5)
            
        with col2:
            st.markdown("##### 📐 Size & Structure")
            living_sqft = st.number_input("Living Area (sqft)", min_value=400, max_value=8000, value=2200, step=50)
            lot_sqft = st.number_input("Lot Size (sqft)", min_value=0, max_value=60000, value=7500, step=250)
            bedrooms = st.slider("Bedrooms", 1, 7, 3)
            bathrooms = st.slider("Bathrooms", 1.0, 6.0, 2.5, 0.5)
            stories = st.selectbox("Stories", options=[1, 2, 3], index=1)
            year_built = st.slider("Year Built", 1945, 2024, 2005)
            property_age = 2026 - year_built
            has_renov = st.selectbox("Been Renovated?", ["No", "Yes"], index=0)
            renovation_age = st.slider("Years Since Renovation", 0, 50, 5) if has_renov == "Yes" else property_age

        with col3:
            st.markdown("##### 🌟 Amenities & Neighborhood")
            has_garage = st.selectbox("Has Garage?", ["Yes", "No"], index=0)
            garage_spaces = st.slider("Garage Spaces", 0, 4, 2 if has_garage == "Yes" else 0)
            has_pool = st.selectbox("Has Pool?", ["No", "Yes"], index=0)
            has_basement = st.selectbox("Has Basement?", ["No", "Yes"], index=0)
            basement_sqft = st.number_input("Basement (sqft)", 0, 4000, 800 if has_basement == "Yes" else 0)
            school_rating = st.slider("School Rating (1-10)", 1, 10, 7)
            safety_score = st.slider("Safety Score (1-100)", 10.0, 100.0, 75.0, 1.0)
            walk_score = st.slider("Walk Score (1-100)", 5.0, 100.0, 60.0, 1.0)
            transit_score = st.slider("Transit Score (1-100)", 5.0, 100.0, 55.0, 1.0)

        # Assemble full input dict with realistic defaults for any extra model features
        prop_input = {
            "city": city,
            "state": state,
            "latitude": CITIES_PROFILE[city]["lat"],
            "longitude": CITIES_PROFILE[city]["lon"],
            "distance_to_downtown_miles": dist_downtown,
            "property_type": prop_type,
            "living_area_sqft": living_sqft,
            "lot_size_sqft": lot_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "rooms_total": bedrooms + int(bathrooms) + 2,
            "year_built": year_built,
            "property_age": property_age,
            "renovation_age": renovation_age,
            "has_been_renovated": has_renov,
            "property_condition": prop_cond,
            "roof_type": "Asphalt Shingle",
            "heating_type": "Central Gas",
            "has_garage": has_garage,
            "garage_spaces": garage_spaces,
            "has_pool": has_pool,
            "has_basement": has_basement,
            "basement_sqft": basement_sqft,
            "has_garden": "Yes" if prop_type == "Single Family" else "No",
            "has_fireplace": "Yes",
            "has_central_ac": "Yes",
            "has_solar_panels": "No",
            "school_rating": school_rating,
            "safety_score": safety_score,
            "walk_score": walk_score,
            "transit_score": transit_score,
            "park_proximity_miles": 0.8,
            "neighborhood_median_income": CITIES_PROFILE[city]["income_mean"],
            "local_property_tax_rate": CITIES_PROFILE[city]["tax_rate"],
            "hoa_fee_monthly": 45.0,
            "days_on_market": 28,
            "mortgage_rate_at_listing": 6.5,
            "energy_efficiency_rating": 7
        }

        # Make prediction
        input_df = pd.DataFrame([prop_input])
        pred_val = pipeline.predict(input_df)[0]
        
        st.markdown("---")
        st.markdown(
            f"""
            <div style="background-color: #f0f4f9; padding: 20px; border-radius: 10px; border-left: 6px solid #1f77b4;">
                <h3 style="margin: 0; color: #1a202c;">Estimated Market Value:</h3>
                <h1 style="margin: 5px 0 0 0; color: #1f77b4; font-size: 2.6rem;">${pred_val:,.0f}</h1>
                <p style="margin: 5px 0 0 0; color: #4a5568;">Based on trained RandomForestRegressor ({bundle['params']['n_estimators']} trees)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("2. Numerical Sensitivity Analysis (What-If Tool)")
        st.caption("Inspect how changing one specific feature influences the valuation, keeping all other features fixed.")
        
        # Choose numerical feature for sensitivity
        num_candidates = [f for f in trained_num if f in [
            "living_area_sqft", "lot_size_sqft", "school_rating", "safety_score",
            "distance_to_downtown_miles", "walk_score", "neighborhood_median_income", "property_age"
        ]]
        if not num_candidates:
            num_candidates = trained_num
            
        sens_feature = st.selectbox("Select Numerical Feature for Sensitivity Analysis:", options=num_candidates)
        
        # Establish realistic ranges
        min_v = float(df[sens_feature].quantile(0.02))
        max_v = float(df[sens_feature].quantile(0.98))
        curr_v = float(prop_input.get(sens_feature, (min_v + max_v) / 2))
        
        df_sens = compute_numerical_sensitivity(
            pipeline=pipeline,
            base_property=prop_input,
            feature_name=sens_feature,
            min_val=min_v,
            max_val=max_v,
            steps=35
        )
        
        fig_sens = plot_sensitivity_curve(df_sens, sens_feature, curr_v, pred_val)
        st.plotly_chart(fig_sens, use_container_width=True)

    # =========================================================================
    # 2. BATCH PREDICTIONS & EXPORT
    # =========================================================================
    elif mode == "📁 Batch Predictions & Export":
        st.subheader("Batch Predictions on Properties")
        st.markdown(
            """
            Upload a CSV file containing property records or use a sample from the existing dataset.
            The model will generate predictions and allow you to **download the enriched CSV**.
            """
        )
        
        source = st.radio("Choose Batch Source:", ["Use Sample from Dataset", "Upload Custom CSV"], horizontal=True)
        
        batch_df = None
        if source == "Use Sample from Dataset":
            sample_size = st.slider("Select sample size", 100, 5000, 1000, 100)
            batch_df = df.sample(sample_size, random_state=42).copy()
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                
        if batch_df is not None:
            st.write(f"Loaded **{len(batch_df):,}** records for batch prediction.")
            
            if st.button("⚡ Generate Batch Predictions", type="primary"):
                with st.spinner("Computing predictions..."):
                    preds = pipeline.predict(batch_df)
                    batch_df["Predicted_Price"] = np.round(preds, 0)
                    
                    if TARGET_COL in batch_df.columns:
                        batch_df["Actual_Price"] = batch_df[TARGET_COL]
                        batch_df["Absolute_Error"] = np.abs(batch_df["Actual_Price"] - batch_df["Predicted_Price"])
                        batch_df["Percentage_Error"] = np.round((batch_df["Absolute_Error"] / batch_df["Actual_Price"]) * 100.0, 2)
                        
                    st.session_state["batch_results"] = batch_df
                    st.success("Batch predictions computed successfully!")
                    
            if "batch_results" in st.session_state:
                res_df = st.session_state["batch_results"]
                st.dataframe(res_df.head(100), use_container_width=True)
                
                # CSV Export
                csv_buffer = io.StringIO()
                res_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Export Predictions to CSV",
                    data=csv_buffer.getvalue(),
                    file_name="real_estate_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

    # =========================================================================
    # 3. CATEGORICAL AGGREGATIONS & ANALYTICS
    # =========================================================================
    elif mode == "📊 Categorical Aggregations & Analytics":
        st.subheader("Categorical Aggregation & Price Discrepancy Analytics")
        st.markdown(
            """
            Group predictions and actual prices by categorical features (**city, state, property_type, condition**).
            Review record counts, medians, means, and model errors per subgroup.
            """
        )
        
        eval_df = bundle.get("eval_sample")
        if eval_df is None:
            # Generate evaluation sample if missing
            sample_eval = df.sample(min(3000, len(df)), random_state=42).copy()
            preds = pipeline.predict(sample_eval)
            sample_eval["Actual_Price"] = sample_eval[TARGET_COL]
            sample_eval["Predicted_Price"] = np.round(preds, 0)
            eval_df = sample_eval
            
        group_col = st.selectbox(
            "Select Categorical Variable for Aggregation:",
            options=["city", "state", "property_type", "property_condition", "has_garage", "has_pool", "has_basement"]
        )
        
        agg_df = compute_categorical_aggregation(eval_df, group_col=group_col)
        
        col_t, col_p = st.columns([1.1, 1])
        with col_t:
            st.dataframe(agg_df, use_container_width=True)
            
            # Export CSV button
            csv_buff = io.StringIO()
            agg_df.to_csv(csv_buff, index=False)
            st.download_button(
                label=f"📥 Export Aggregations ({group_col}) to CSV",
                data=csv_buff.getvalue(),
                file_name=f"aggregation_{group_col}.csv",
                mime="text/csv"
            )
            
        with col_p:
            fig_bar = plot_categorical_comparison(agg_df, group_col)
            st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================================
    # 4. ANALYTICS FEATURE EXPLORATION
    # =========================================================================
    elif mode == "🔍 Analytics Feature Exploration":
        st.subheader("Analytics Feature Selection (Explore Subsets)")
        st.caption("Filter test samples by analytics criteria without modifying or retraining the active ML model.")
        
        eval_df = bundle.get("eval_sample")
        if eval_df is None:
            eval_df = df.copy()
            eval_df["Actual_Price"] = eval_df[TARGET_COL]
            eval_df["Predicted_Price"] = pipeline.predict(eval_df)
            eval_df["Residual"] = eval_df["Actual_Price"] - eval_df["Predicted_Price"]
            eval_df["Abs_Error"] = np.abs(eval_df["Residual"])
            eval_df["Pct_Error"] = (eval_df["Abs_Error"] / eval_df["Actual_Price"]) * 100.0

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sel_cities = st.multiselect("Filter Cities:", options=sorted(eval_df["city"].unique()), default=sorted(eval_df["city"].unique())[:3])
        with f_col2:
            sel_types = st.multiselect("Filter Property Types:", options=sorted(eval_df["property_type"].unique()), default=sorted(eval_df["property_type"].unique()))
        with f_col3:
            price_range = st.slider(
                "Filter Price Range ($):",
                min_value=int(eval_df["Actual_Price"].min()),
                max_value=int(eval_df["Actual_Price"].max()),
                value=(int(eval_df["Actual_Price"].quantile(0.05)), int(eval_df["Actual_Price"].quantile(0.95))),
                step=25000
            )
            
        filtered = eval_df[
            (eval_df["city"].isin(sel_cities)) &
            (eval_df["property_type"].isin(sel_types)) &
            (eval_df["Actual_Price"].between(price_range[0], price_range[1]))
        ]
        
        st.write(f"Filtered subset contains **{len(filtered):,}** records.")
        if len(filtered) > 0:
            fig_flt = plot_actual_vs_predicted(filtered)
            st.plotly_chart(fig_flt, use_container_width=True)
            
            sub_m1, sub_m2, sub_m3 = st.columns(3)
            sub_m1.metric("Subset MAE", f"${filtered['Abs_Error'].mean():,.0f}")
            sub_m2.metric("Subset Median Error", f"${filtered['Abs_Error'].median():,.0f}")
            sub_m3.metric("Subset Mean MAPE", f"{filtered['Pct_Error'].mean():.2f}%")


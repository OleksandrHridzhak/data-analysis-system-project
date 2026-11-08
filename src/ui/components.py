import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_actual_vs_predicted(df: pd.DataFrame, sample_size: int = 2000) -> go.Figure:
    """
    Interactive Plotly scatter plot of Actual vs Predicted prices with ideal y=x reference line.
    """
    plot_df = df.sample(min(len(df), sample_size), random_state=42) if len(df) > sample_size else df
    
    fig = px.scatter(
        plot_df,
        x="Actual_Price",
        y="Predicted_Price",
        color="Pct_Error",
        color_continuous_scale="Viridis",
        labels={
            "Actual_Price": "Actual Price ($)",
            "Predicted_Price": "Predicted Price ($)",
            "Pct_Error": "Error (%)"
        },
        title=f"Actual vs. Predicted Real Estate Prices (Sample of {len(plot_df):,} properties)",
        hover_data=["city", "property_type", "living_area_sqft"] if "city" in plot_df.columns else None,
        opacity=0.65
    )
    
    # Ideal diagonal line
    min_val = min(plot_df["Actual_Price"].min(), plot_df["Predicted_Price"].min())
    max_val = max(plot_df["Actual_Price"].max(), plot_df["Predicted_Price"].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Prediction (y = x)",
            line=dict(color="red", dash="dash", width=2)
        )
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(tickformat="$,.0f"),
        yaxis=dict(tickformat="$,.0f"),
        height=520,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_residuals(df: pd.DataFrame, sample_size: int = 2000) -> go.Figure:
    """
    Plotly figure for Residual Distribution and Residuals vs Predicted.
    """
    plot_df = df.sample(min(len(df), sample_size), random_state=42) if len(df) > sample_size else df
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=plot_df["Predicted_Price"],
            y=plot_df["Residual"],
            mode="markers",
            marker=dict(
                size=5,
                color=plot_df["Residual"],
                colorscale="RdBu",
                showscale=True,
                colorbar=dict(title="Residual ($)")
            ),
            name="Residuals",
            opacity=0.65
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5)
    
    fig.update_layout(
        title="Residuals vs. Predicted Price (Checking Homoscedasticity)",
        xaxis_title="Predicted Price ($)",
        yaxis_title="Residual (Actual - Predicted) ($)",
        xaxis=dict(tickformat="$,.0f"),
        yaxis=dict(tickformat="$,.0f"),
        template="plotly_white",
        height=480
    )
    return fig


def plot_feature_importance(df_importance: pd.DataFrame) -> go.Figure:
    """
    Interactive horizontal bar chart of top feature importances.
    """
    sorted_df = df_importance.sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        sorted_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance (MDI)",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        yaxis=dict(title=""),
        xaxis=dict(title="Relative Importance Score"),
        coloraxis_showscale=False
    )
    return fig


def plot_sensitivity_curve(df_sens: pd.DataFrame, feature_name: str, base_val: float, base_pred: float) -> go.Figure:
    """
    Plots the sensitivity curve as one feature varies.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df_sens[feature_name],
            y=df_sens["Predicted_Price"],
            mode="lines+markers",
            name="Predicted Price",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=4)
        )
    )
    
    # Mark current base value
    fig.add_trace(
        go.Scatter(
            x=[base_val],
            y=[base_pred],
            mode="markers",
            name="Current Base Property",
            marker=dict(color="red", size=12, symbol="star")
        )
    )
    
    fig.update_layout(
        title=f"Price Sensitivity to '{feature_name}' (Other Features Fixed)",
        xaxis_title=feature_name,
        yaxis_title="Predicted Price ($)",
        yaxis=dict(tickformat="$,.0f"),
        template="plotly_white",
        height=460,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_categorical_comparison(df_cat: pd.DataFrame, cat_col: str) -> go.Figure:
    """
    Grouped bar chart comparing Mean Actual vs Mean Predicted price across categories.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=df_cat[cat_col],
            y=df_cat["Mean Actual ($)"],
            name="Mean Actual Price",
            marker_color="#2ca02c"
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=df_cat[cat_col],
            y=df_cat["Mean Predicted ($)"],
            name="Mean Predicted Price",
            marker_color="#1f77b4"
        )
    )
    
    fig.update_layout(
        title=f"Price Comparison across Categories: {cat_col}",
        barmode="group",
        xaxis_title=cat_col,
        yaxis_title="Price ($)",
        yaxis=dict(tickformat="$,.0f"),
        template="plotly_white",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig


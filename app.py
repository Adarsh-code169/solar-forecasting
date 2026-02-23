import os
import sys
import numpy as np
import pandas as pd
import streamlit as st


sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_csv, load_uploaded_file
from src.preprocessing import clean_data, add_time_features, prepare_data, TARGET
from src.model import (
    train_linear_regression,
    train_random_forest,
    predict,
    evaluate_model,
)
from src.visualization import (
    plot_power_trend,
    plot_hourly_pattern,
    plot_seasonal_pattern,
    plot_actual_vs_predicted,
    plot_scatter_actual_vs_predicted,
    plot_feature_importance,
    plot_correlation_heatmap,
)

# Page config
st.set_page_config(
    page_title="Solar Energy Forecasting",
    page_icon="☀️",
    layout="wide",
)

# Sidebar data source
st.sidebar.title("☀️ Solar Forecasting")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data source",
    ["Default dataset", "Upload CSV"],
)

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "solar_data.csv")

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        raw_df = load_uploaded_file(uploaded)
    else:
        st.info("⬆️ Please upload a CSV file to get started.")
        st.stop()
else:
    raw_df = load_csv(DEFAULT_CSV)

# Preprocess once and cache
df_clean = clean_data(raw_df)
df_featured = add_time_features(df_clean)

# Tabs
tab1, tab2, tab3 = st.tabs(["Data Explorer", "Trends & Seasonality", "Forecasting"])



# TAB 1 – Data Explorer
with tab1:
    st.header("Data Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df_featured))
    col2.metric("Total Columns", len(df_featured.columns))
    col3.metric("Avg Power (kW)", round(df_featured[TARGET].mean(), 2))

    st.subheader("Raw Data Preview")
    st.dataframe(df_featured.head(100), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df_featured.describe().round(2), use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig_corr = plot_correlation_heatmap(df_featured)
    st.plotly_chart(fig_corr, use_container_width=True)


# TAB 2 – Trends & Seasonality
with tab2:
    st.header("Trends & Seasonality Analysis")

    st.subheader("Power Generation Trend")
    fig_trend = plot_power_trend(df_featured)
    st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Hourly Pattern")
        fig_hour = plot_hourly_pattern(df_featured)
        st.plotly_chart(fig_hour, use_container_width=True)

    with col_b:
        st.subheader("Seasonal Pattern")
        fig_season = plot_seasonal_pattern(df_featured)
        st.plotly_chart(fig_season, use_container_width=True)

# TAB 3 – Forecasting
with tab3:
    st.header("Solar Power Forecasting")

    # Model selection
    model_choice = st.selectbox(
        "Select model",
        ["Linear Regression", "Random Forest"],
    )

    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100.0

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Preparing data..."):
            X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
                raw_df, test_size=test_size
            )

        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Linear Regression":
                model = train_linear_regression(X_train, y_train)
            else:
                model = train_random_forest(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        y_pred = predict(model, X_test)

        # Metrics
        st.subheader("Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['MAE']:.2f} kW")
        m2.metric("RMSE", f"{metrics['RMSE']:.2f} kW")
        m3.metric("R² Score", f"{metrics['R2']:.4f}")

        # Actual vs Predicted charts
        st.subheader("Actual vs Predicted")
        fig_avp = plot_actual_vs_predicted(
            y_test, y_pred, title=f"{model_choice} — Actual vs Predicted"
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        st.subheader("Prediction Accuracy Scatter")
        fig_scatter = plot_scatter_actual_vs_predicted(
            y_test, y_pred, title=f"{model_choice} — Scatter"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Feature Importance (RF only)
        if model_choice == "Random Forest":
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            fig_imp = plot_feature_importance(importances, feature_names)
            st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Solar Energy Forecasting — Milestone 1")

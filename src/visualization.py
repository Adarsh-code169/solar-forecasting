import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Trend
def plot_power_trend(df: pd.DataFrame):
    # Line plot of solar power generation over time
    fig = px.line(
        df.reset_index(),
        x="index",
        y="generated_power_kw",
        title="Solar Power Generation Trend",
        labels={"index": "Time Step", "generated_power_kw": "Power (kW)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


# Seasonality

def plot_hourly_pattern(df: pd.DataFrame):
    """Box plot of power generation grouped by hour_of_day."""
    if "hour_of_day" not in df.columns:
        return _empty_figure("hour_of_day column missing")

    fig = px.box(
        df,
        x="hour_of_day",
        y="generated_power_kw",
        title="Power Generation by Hour of Day",
        labels={"hour_of_day": "Hour of Day", "generated_power_kw": "Power (kW)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_seasonal_pattern(df: pd.DataFrame):
    """Box plot of power generation grouped by season."""
    if "season" not in df.columns:
        return _empty_figure("season column missing")

    season_map = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    temp = df.copy()
    temp["season_name"] = temp["season"].map(season_map)

    fig = px.box(
        temp,
        x="season_name",
        y="generated_power_kw",
        title="Power Generation by Season",
        labels={"season_name": "Season", "generated_power_kw": "Power (kW)"},
        category_orders={"season_name": ["Winter", "Spring", "Summer", "Autumn"]},
    )
    fig.update_layout(template="plotly_white")
    return fig





#Feature Importance

def plot_feature_importance(importances, feature_names):
    """Horizontal bar chart of feature importances (e.g. from Random Forest)."""
    sorted_idx = np.argsort(importances)
    fig = go.Figure(
        go.Bar(
            x=importances[sorted_idx],
            y=np.array(feature_names)[sorted_idx],
            orientation="h",
            marker_color="#636EFA",
        )
    )
    fig.update_layout(
        title="Feature Importance (Random Forest)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, len(feature_names) * 25),
    )
    return fig


# Correlation Heatmap

def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap of Pearson correlations among numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template="plotly_white",
        height=700,
        width=800,
    )
    return fig


# Actual vs Predicted (Milestone 1 — evaluation of forecasting accuracy)

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """Line chart comparing actual and predicted solar power values."""
    import pandas as pd
    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["Actual"], mode="lines", name="Actual", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(y=df["Predicted"], mode="lines", name="Predicted", line=dict(color="#EF553B", dash="dash")))
    fig.update_layout(
        title=title,
        xaxis_title="Sample",
        yaxis_title="Power (kW)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_scatter_actual_vs_predicted(y_true, y_pred, title="Scatter: Actual vs Predicted"):
    """Scatter plot of actual vs predicted values with a perfect-prediction reference line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color="#00CC96", opacity=0.6, size=5),
        name="Predictions",
    ))
    min_val = float(min(min(y_true), min(y_pred)))
    max_val = float(max(max(y_true), max(y_pred)))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Perfect Fit",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Actual (kW)",
        yaxis_title="Predicted (kW)",
        template="plotly_white",
    )
    return fig


#Helpers 

def _empty_figure(message: str):
    """Return a blank figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(template="plotly_white")
    return fig
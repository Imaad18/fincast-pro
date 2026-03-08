"""
FinCast Pro — Financial Forecasting Platform
For Finance Analysts / FP&A Teams
Author: Built with Streamlit + Prophet + ARIMA + XGBoost + Monte Carlo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import io
import base64
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinCast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Professional Theme ────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg-primary:    #09090f;
    --bg-card:       #111118;
    --bg-elevated:   #16161f;
    --border:        #1e1e2e;
    --accent:        #00d4ff;
    --accent-dim:    rgba(0,212,255,0.12);
    --accent-glow:   rgba(0,212,255,0.25);
    --green:         #00ff87;
    --red:           #ff4757;
    --yellow:        #ffd32a;
    --text-primary:  #e8e8f0;
    --text-secondary:#8888aa;
    --text-muted:    #44445a;
  }

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  /* Main */
  .main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  /* Metric cards */
  .metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
  }
  .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
  }
  .metric-delta-pos { color: var(--green); font-size: 0.82rem; margin-top: 0.3rem; }
  .metric-delta-neg { color: var(--red);   font-size: 0.82rem; margin-top: 0.3rem; }

  /* Section headers */
  .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 1.8rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* Logo / title bar */
  .app-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
  }
  .app-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
  }
  .app-title span { color: var(--accent); }
  .app-subtitle {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-weight: 400;
  }

  /* Status badges */
  .badge-success {
    background: rgba(0,255,135,0.12);
    color: var(--green);
    border: 1px solid rgba(0,255,135,0.25);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
  }
  .badge-info {
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-glow);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
  }
  .badge-warn {
    background: rgba(255,211,42,0.12);
    color: var(--yellow);
    border: 1px solid rgba(255,211,42,0.25);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
  }

  /* Model comparison table */
  .model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 0.5rem;
  }
  .model-table th {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  .model-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }
  .model-table tr:last-child td { border-bottom: none; }
  .model-table tr.best-model td { background: var(--accent-dim); }
  .model-table tr:hover td { background: var(--bg-elevated); }
  .best-label {
    background: var(--accent);
    color: #000;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.65rem;
    font-weight: 700;
    margin-left: 6px;
  }

  /* Insight boxes */
  .insight-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-size: 0.875rem;
    color: var(--text-primary);
    line-height: 1.6;
  }

  /* Streamlit overrides */
  .stSelectbox > div > div, .stMultiSelect > div > div {
    background-color: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
  }
  .stSlider > div > div > div { background: var(--accent) !important; }
  .stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  div[data-testid="stFileUploader"] {
    background: var(--bg-elevated) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
  }

  div.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0;
  }
  div.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.25rem !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
  }
  div.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: var(--accent-dim) !important;
  }

  div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
  }

  .stDataFrame { background: var(--bg-card) !important; }

  .stAlert {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
  }

  /* Anomaly callout */
  .anomaly-row {
    background: rgba(255,71,87,0.08);
    border-left: 3px solid var(--red);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    margin: 0.2rem 0;
    color: var(--text-primary);
  }

  /* Hide default streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .viewerBadge_container__1QSob { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Plotly default theme ───────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#09090f",
    plot_bgcolor="#09090f",
    font=dict(family="Space Grotesk", color="#8888aa", size=12),
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zeroline=False, tickfont=dict(color="#8888aa")),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zeroline=False, tickfont=dict(color="#8888aa")),
    legend=dict(bgcolor="#111118", bordercolor="#1e1e2e", borderwidth=1,
                font=dict(color="#8888aa", size=11)),
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#16161f", bordercolor="#1e1e2e",
                    font=dict(color="#e8e8f0", size=12)),
)

COLORS = {
    "actual":    "#8888aa",
    "prophet":   "#00d4ff",
    "arima":     "#00ff87",
    "xgb":       "#ffd32a",
    "monte":     "#ff6b9d",
    "best":      "#00d4ff",
    "worst":     "#ff4757",
    "ci_upper":  "rgba(0,212,255,0.15)",
    "ci_lower":  "rgba(0,212,255,0.05)",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample_revenue(n=36):
    """Realistic monthly revenue with trend + seasonality + noise."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today().replace(day=1), periods=n, freq="MS")
    trend  = np.linspace(500_000, 820_000, n)
    season = 60_000 * np.sin(2 * np.pi * np.arange(n) / 12 - np.pi / 3)
    noise  = np.random.normal(0, 18_000, n)
    values = trend + season + noise
    values = np.clip(values, 0, None)
    return pd.DataFrame({"ds": dates, "y": values.round(2)})


def generate_sample_stock(n=500):
    """Geometric Brownian Motion for stock price."""
    np.random.seed(7)
    dates  = pd.date_range(end=datetime.today(), periods=n, freq="B")
    drift  = 0.0004
    vol    = 0.016
    shocks = np.random.normal(drift, vol, n)
    price  = 100 * np.exp(np.cumsum(shocks))
    return pd.DataFrame({"ds": dates, "y": price.round(4)})


def detect_anomalies_iqr(series: pd.Series, factor=2.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr     = q3 - q1
    lo, hi  = q1 - factor * iqr, q3 + factor * iqr
    return (series < lo) | (series > hi)


def smape(actual, forecast):
    a, f = np.array(actual), np.array(forecast)
    mask = ~(np.isnan(a) | np.isnan(f))
    return 100 * np.mean(2 * np.abs(f[mask] - a[mask]) / (np.abs(a[mask]) + np.abs(f[mask]) + 1e-9))


def rmse(actual, forecast):
    a, f = np.array(actual), np.array(forecast)
    mask = ~(np.isnan(a) | np.isnan(f))
    return np.sqrt(np.mean((a[mask] - f[mask]) ** 2))


def mape(actual, forecast):
    a, f = np.array(actual), np.array(forecast)
    mask = (a != 0) & ~np.isnan(a) & ~np.isnan(f)
    return 100 * np.mean(np.abs((a[mask] - f[mask]) / a[mask]))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS  — each returns (forecast_df, in_sample_pred)
# ══════════════════════════════════════════════════════════════════════════════

def run_prophet(df: pd.DataFrame, horizon: int, freq: str = "MS",
                yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False):
    try:
        from prophet import Prophet
        m = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.9,
            changepoint_prior_scale=0.05,
        )
        m.fit(df[["ds", "y"]])
        future = m.make_future_dataframe(periods=horizon, freq=freq)
        fc     = m.predict(future)
        fwd    = fc[fc["ds"] > df["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        insample = fc[fc["ds"] <= df["ds"].max()][["ds", "yhat"]].rename(columns={"yhat": "y_pred"})
        fwd.columns = ["ds", "yhat", "lower", "upper"]
        return fwd, insample
    except ImportError:
        return None, None


def run_arima(df: pd.DataFrame, horizon: int, freq: str = "MS"):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        series = df.set_index("ds")["y"]
        # Auto-detect seasonality period
        s = 12 if freq in ("MS", "M", "ME") else (5 if freq == "B" else 7)
        order        = (1, 1, 1)
        seasonal_order = (1, 1, 0, s) if len(series) >= 2 * s else (0, 0, 0, 0)
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res  = model.fit(disp=False, maxiter=200)
        fc   = res.get_forecast(steps=horizon)
        fwd_mean = fc.predicted_mean
        ci        = fc.conf_int(alpha=0.10)
        fwd_dates = pd.date_range(start=df["ds"].max() + pd.tseries.frequencies.to_offset(freq),
                                  periods=horizon, freq=freq)
        fwd = pd.DataFrame({
            "ds":    fwd_dates,
            "yhat":  fwd_mean.values,
            "lower": ci.iloc[:, 0].values,
            "upper": ci.iloc[:, 1].values,
        })
        in_sample_pred = pd.DataFrame({
            "ds":     series.index,
            "y_pred": res.fittedvalues.values,
        })
        return fwd, in_sample_pred
    except Exception:
        return None, None


def run_xgboost(df: pd.DataFrame, horizon: int, freq: str = "MS", n_lags: int = 12):
    try:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        series = df["y"].values.astype(float)
        n      = len(series)
        # Feature engineering
        def make_features(s, lags):
            X, y_arr = [], []
            for i in range(lags, len(s)):
                row = list(s[i - lags:i])
                # rolling stats
                row += [np.mean(s[i - lags:i]), np.std(s[i - lags:i]),
                        np.min(s[i - lags:i]),  np.max(s[i - lags:i])]
                X.append(row); y_arr.append(s[i])
            return np.array(X), np.array(y_arr)

        X, y_arr = make_features(series, n_lags)
        split = max(1, int(len(X) * 0.2))
        X_tr, y_tr = X[:-split], y_arr[:-split]
        X_val, y_val = X[-split:], y_arr[-split:]

        model = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  verbose=False)

        # In-sample predictions
        X_all, y_all = make_features(series, n_lags)
        in_sample_vals = model.predict(X_all)
        in_sample_pred = pd.DataFrame({
            "ds":     df["ds"].iloc[n_lags:].values,
            "y_pred": in_sample_vals,
        })

        # Recursive multi-step forecast with MC uncertainty
        last_window = list(series[-n_lags:])
        forecasts, lower, upper = [], [], []
        for _ in range(horizon):
            row = np.array(last_window[-n_lags:])
            feats = np.concatenate([row, [np.mean(row), np.std(row),
                                          np.min(row), np.max(row)]])
            pred  = float(model.predict(feats.reshape(1, -1))[0])
            # Bootstrap uncertainty
            noise_std = np.std(series) * 0.05
            forecasts.append(pred)
            lower.append(pred - 1.645 * noise_std * np.sqrt(_ + 1))
            upper.append(pred + 1.645 * noise_std * np.sqrt(_ + 1))
            last_window.append(pred)

        fwd_dates = pd.date_range(
            start=df["ds"].max() + pd.tseries.frequencies.to_offset(freq),
            periods=horizon, freq=freq,
        )
        fwd = pd.DataFrame({
            "ds":    fwd_dates,
            "yhat":  forecasts,
            "lower": lower,
            "upper": upper,
        })
        return fwd, in_sample_pred
    except Exception as e:
        return None, None


def run_monte_carlo(df: pd.DataFrame, horizon: int, freq: str = "MS",
                    n_simulations: int = 1000, scenario: str = "base"):
    """GBM-based Monte Carlo with scenario adjustments."""
    series  = df["y"].values.astype(float)
    log_ret = np.diff(np.log(series + 1e-9))
    mu      = np.mean(log_ret)
    sigma   = np.std(log_ret)

    # Scenario tweaks
    scen_mu = {"best": mu * 1.4, "base": mu, "worst": mu * 0.4}
    scen_sig = {"best": sigma * 0.7, "base": sigma, "worst": sigma * 1.4}
    mu_use    = scen_mu.get(scenario, mu)
    sigma_use = scen_sig.get(scenario, sigma)

    np.random.seed(123)
    last_val = series[-1]
    paths = np.zeros((n_simulations, horizon))
    for i in range(n_simulations):
        shocks = np.random.normal(mu_use, sigma_use, horizon)
        path   = last_val * np.exp(np.cumsum(shocks))
        paths[i] = path

    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    fwd_dates = pd.date_range(
        start=df["ds"].max() + pd.tseries.frequencies.to_offset(freq),
        periods=horizon, freq=freq,
    )
    fwd = pd.DataFrame({
        "ds":    fwd_dates,
        "yhat":  p50,
        "lower": p10,
        "upper": p90,
    })
    return fwd, paths


# ══════════════════════════════════════════════════════════════════════════════
# CHARTING
# ══════════════════════════════════════════════════════════════════════════════

def chart_historical(df: pd.DataFrame, anomalies: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"], mode="lines",
        line=dict(color=COLORS["actual"], width=1.8),
        name="Historical", fill="tozeroy",
        fillcolor="rgba(136,136,170,0.06)",
    ))
    anom_df = df[anomalies]
    if not anom_df.empty:
        fig.add_trace(go.Scatter(
            x=anom_df["ds"], y=anom_df["y"], mode="markers",
            marker=dict(color="#ff4757", size=9, symbol="circle-open",
                        line=dict(width=2, color="#ff4757")),
            name="Anomaly",
        ))
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(color="#e8e8f0", size=14), x=0.01))
    return fig


def chart_forecast(df: pd.DataFrame, forecasts: dict, title: str) -> go.Figure:
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"], mode="lines",
        line=dict(color=COLORS["actual"], width=1.8),
        name="Historical",
    ))
    # Vertical divider
    split = df["ds"].max()
    fig.add_vline(x=split, line_dash="dot", line_color="#44445a", line_width=1.5)

    model_color = {"Prophet": COLORS["prophet"], "ARIMA": COLORS["arima"],
                   "XGBoost": COLORS["xgb"],    "Monte Carlo": COLORS["monte"]}

    for name, fwd in forecasts.items():
        if fwd is None: continue
        col = model_color.get(name, "#ffffff")
        # CI fill
        fig.add_trace(go.Scatter(
            x=pd.concat([fwd["ds"], fwd["ds"][::-1]]),
            y=pd.concat([fwd["upper"], fwd["lower"][::-1]]),
            fill="toself",
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fwd["ds"], y=fwd["yhat"], mode="lines",
            line=dict(color=col, width=2, dash="dot" if name == "ARIMA" else "solid"),
            name=name,
        ))

    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(color="#e8e8f0", size=14), x=0.01))
    return fig


def chart_monte_carlo_fan(df: pd.DataFrame, paths: np.ndarray, fwd_dates, n_show=100) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"], mode="lines",
        line=dict(color=COLORS["actual"], width=2),
        name="Historical",
    ))
    idx = np.random.choice(paths.shape[0], min(n_show, paths.shape[0]), replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=fwd_dates, y=paths[i], mode="lines",
            line=dict(color="rgba(255,107,157,0.07)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Percentile bands
    for p, label, col in [(90, "P90", "#ff4757"), (50, "Median", "#ff6b9d"), (10, "P10", "#00ff87")]:
        vals = np.percentile(paths, p, axis=0)
        fig.add_trace(go.Scatter(
            x=fwd_dates, y=vals, mode="lines",
            line=dict(color=col, width=2.5),
            name=label,
        ))
    fig.add_vline(x=df["ds"].max(), line_dash="dot", line_color="#44445a", line_width=1.5)
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="Monte Carlo Fan Chart — Simulated Paths",
                                 font=dict(color="#e8e8f0", size=14), x=0.01))
    return fig


def chart_returns_distribution(series: pd.Series) -> go.Figure:
    rets = series.pct_change().dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=50,
        marker_color=COLORS["prophet"],
        marker_line=dict(width=0.5, color="#09090f"),
        name="Returns (%)",
    ))
    mean_r = rets.mean()
    fig.add_vline(x=mean_r, line_dash="dash", line_color="#ffd32a", line_width=1.5,
                  annotation_text=f"Mean {mean_r:.2f}%",
                  annotation_font_color="#ffd32a",
                  annotation_position="top right")
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="Return Distribution", font=dict(color="#e8e8f0", size=14), x=0.01),
                      showlegend=False)
    return fig


def chart_model_errors(scores: dict) -> go.Figure:
    if not scores: return go.Figure()
    models = list(scores.keys())
    mapes  = [scores[m]["MAPE"] for m in models]
    rmses  = [scores[m]["RMSE"] for m in models]
    colors_bar = [COLORS.get(m.lower().replace(" ", ""), "#8888aa") for m in models]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["MAPE (%) — Lower is Better",
                                        "RMSE — Lower is Better"])
    fig.add_trace(go.Bar(x=models, y=mapes, marker_color=colors_bar, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=rmses, marker_color=colors_bar, showlegend=False), row=1, col=2)
    fig.update_layout(**CHART_LAYOUT,
                      paper_bgcolor="#09090f", plot_bgcolor="#09090f",
                      title=dict(text="Model Accuracy Comparison (In-Sample)",
                                 font=dict(color="#e8e8f0", size=14), x=0.01))
    for ann in fig.layout.annotations:
        ann.font.color = "#8888aa"
        ann.font.size  = 11
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def to_excel_bytes(dfs: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1.5rem;">
          <div style="font-size:1.2rem;font-weight:700;color:#e8e8f0;letter-spacing:-0.02em;">
            📈 <span style="color:#00d4ff;">FinCast</span> Pro
          </div>
          <div style="font-size:0.7rem;color:#8888aa;margin-top:0.2rem;letter-spacing:0.05em;text-transform:uppercase;">
            FP&A Forecasting Platform
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
        data_mode = st.radio("", ["Upload CSV", "Use Sample Data"], label_visibility="collapsed")

        df = None
        if data_mode == "Upload CSV":
            f = st.file_uploader("Drop your CSV here", type=["csv"],
                                 help="Required columns: ds (date), y (numeric value)")
            if f:
                try:
                    df = pd.read_csv(f, parse_dates=["ds"])
                    df = df.sort_values("ds").reset_index(drop=True)
                    st.markdown(f'<span class="badge-success">✓ {len(df)} rows loaded</span>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Parse error: {e}")
        else:
            asset_type = st.selectbox("Sample dataset", ["Monthly Revenue", "Stock Price"])
            df = generate_sample_revenue() if asset_type == "Monthly Revenue" else generate_sample_stock()
            st.markdown(f'<span class="badge-info">Sample · {len(df)} rows</span>', unsafe_allow_html=True)

        if df is None:
            st.stop()

        st.markdown('<div class="section-header">Forecast Settings</div>', unsafe_allow_html=True)
        freq_map = {"Monthly": "MS", "Weekly": "W", "Daily (Business)": "B", "Quarterly": "QS"}
        freq_label = st.selectbox("Data Frequency", list(freq_map.keys()))
        freq = freq_map[freq_label]
        horizon_label = {"Monthly": "months", "Weekly": "weeks",
                         "Daily (Business)": "business days", "Quarterly": "quarters"}
        max_h = {"Monthly": 24, "Weekly": 52, "Daily (Business)": 252, "Quarterly": 8}
        horizon = st.slider(f"Forecast horizon ({horizon_label[freq_label]})",
                            1, max_h[freq_label],
                            {"Monthly": 12, "Weekly": 26, "Daily (Business)": 60, "Quarterly": 4}[freq_label])

        st.markdown('<div class="section-header">Models</div>', unsafe_allow_html=True)
        run_prophet_m  = st.checkbox("Prophet (Meta)", value=True)
        run_arima_m    = st.checkbox("ARIMA / SARIMA", value=True)
        run_xgb_m      = st.checkbox("XGBoost ML", value=True)
        run_mc_m       = st.checkbox("Monte Carlo", value=True)

        if run_mc_m:
            n_sims = st.slider("MC simulations", 200, 5000, 1000, step=200)
        else:
            n_sims = 1000

        st.markdown('<div class="section-header">Scenario Engine</div>', unsafe_allow_html=True)
        scenario = st.select_slider("", options=["worst", "base", "best"], value="base")
        scenario_color = {"best": "#00ff87", "base": "#00d4ff", "worst": "#ff4757"}[scenario]
        st.markdown(f'<span class="badge-info" style="border-color:{scenario_color};color:{scenario_color};">'
                    f'{scenario.upper()} CASE</span>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
        iqr_factor = st.slider("IQR sensitivity", 1.0, 4.0, 2.0, 0.5)

        st.markdown("---")
        run_btn = st.button("⚡  Run Forecast", use_container_width=True)

        return (df, freq, horizon, run_prophet_m, run_arima_m, run_xgb_m, run_mc_m,
                n_sims, scenario, iqr_factor, run_btn)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    (df, freq, horizon, run_prophet_m, run_arima_m, run_xgb_m, run_mc_m,
     n_sims, scenario, iqr_factor, run_btn) = sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
      <span class="app-title">Fin<span>Cast</span> Pro</span>
      <span class="app-subtitle">Financial Forecasting & Scenario Planning Platform</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    series = df["y"]
    pct_chg = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
    recent_chg = (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] * 100 if len(series) > 1 else 0
    vol = series.pct_change().std() * 100
    anom = detect_anomalies_iqr(series, iqr_factor)

    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, label, value, delta=None, delta_pos=True, prefix="", suffix=""):
        delta_html = ""
        if delta is not None:
            cls = "metric-delta-pos" if delta_pos else "metric-delta-neg"
            arrow = "▲" if delta_pos else "▼"
            delta_html = f'<div class="{cls}">{arrow} {abs(delta):.2f}%</div>'
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{prefix}{value}{suffix}</div>
          {delta_html}
        </div>""", unsafe_allow_html=True)

    with c1: kpi(c1, "Latest Value", f"{series.iloc[-1]:,.0f}")
    with c2: kpi(c2, "Period Change", f"{pct_chg:+.1f}", delta=abs(pct_chg), delta_pos=pct_chg>0, suffix="%")
    with c3: kpi(c3, "Last Δ", f"{recent_chg:+.2f}", delta=abs(recent_chg), delta_pos=recent_chg>0, suffix="%")
    with c4: kpi(c4, "Volatility (σ)", f"{vol:.2f}", suffix="%")
    with c5: kpi(c5, "Anomalies", str(anom.sum()))

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊  Historical Analysis",
        "🔮  Forecast",
        "🎲  Monte Carlo",
        "📐  Model Comparison",
        "📤  Export",
    ])

    # ══ TAB 1 — Historical ═══════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Historical Data & Anomaly Detection</div>', unsafe_allow_html=True)
        fig_hist = chart_historical(df, anom, "Historical Series with Anomaly Overlay")
        st.plotly_chart(fig_hist, use_container_width=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown('<div class="section-header">Return Distribution</div>', unsafe_allow_html=True)
            fig_ret = chart_returns_distribution(series)
            st.plotly_chart(fig_ret, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
            stats = pd.DataFrame({
                "Metric": ["Observations", "Mean", "Median", "Std Dev",
                            "Min", "Max", "Skewness", "Kurtosis"],
                "Value": [
                    f"{len(series):,}",
                    f"{series.mean():,.2f}",
                    f"{series.median():,.2f}",
                    f"{series.std():,.2f}",
                    f"{series.min():,.2f}",
                    f"{series.max():,.2f}",
                    f"{series.skew():.4f}",
                    f"{series.kurt():.4f}",
                ]
            })
            st.dataframe(stats, hide_index=True, use_container_width=True)

            # Anomalies
            if anom.sum() > 0:
                st.markdown('<div class="section-header">Flagged Anomalies</div>', unsafe_allow_html=True)
                for idx_, row in df[anom].iterrows():
                    st.markdown(
                        f'<div class="anomaly-row">📍 {row["ds"].strftime("%Y-%m-%d")} &nbsp;|&nbsp; {row["y"]:,.2f}</div>',
                        unsafe_allow_html=True)

        # Analyst insight
        skew = series.skew()
        insight_text = (
            f"The series spans <b>{len(df)}</b> observations with a mean of <b>{series.mean():,.0f}</b> "
            f"and volatility of <b>{vol:.1f}%</b>. "
            f"{'Positive skew suggests upside tail risk — watch for outlier months.' if skew > 0.5 else 'Negative skew indicates downside tail risk — consider stress testing.' if skew < -0.5 else 'Return distribution is approximately symmetric.'} "
            f"{'<b>' + str(anom.sum()) + ' anomalous observations</b> were detected — review before forecasting.' if anom.sum() > 0 else 'No significant anomalies found — data quality looks clean.'}"
        )
        st.markdown(f'<div class="insight-box">🧠 <b>Analyst Insight:</b><br>{insight_text}</div>',
                    unsafe_allow_html=True)

    # ══ TAB 2 — Forecast ══════════════════════════════════════════════════════
    with tab2:
        if not run_btn:
            st.markdown("""
            <div style="text-align:center;padding:4rem 0;color:#44445a;">
              <div style="font-size:3rem;">⚡</div>
              <div style="font-size:1.1rem;margin-top:1rem;color:#8888aa;">
                Configure your settings in the sidebar and click <b style="color:#00d4ff;">Run Forecast</b>
              </div>
            </div>""", unsafe_allow_html=True)
            return

        forecasts   = {}
        insample    = {}
        mc_paths    = None
        mc_fwd      = None

        with st.spinner("Running models..."):
            if run_prophet_m:
                fwd, ins = run_prophet(df, horizon, freq)
                if fwd is not None:
                    forecasts["Prophet"] = fwd
                    insample["Prophet"]  = ins
                else:
                    st.warning("Prophet not installed. Run: `pip install prophet`")

            if run_arima_m:
                fwd, ins = run_arima(df, horizon, freq)
                if fwd is not None:
                    forecasts["ARIMA"] = fwd
                    insample["ARIMA"]  = ins
                else:
                    st.warning("statsmodels not installed.")

            if run_xgb_m:
                fwd, ins = run_xgboost(df, horizon, freq)
                if fwd is not None:
                    forecasts["XGBoost"] = fwd
                    insample["XGBoost"]  = ins
                else:
                    st.warning("XGBoost not installed.")

            if run_mc_m:
                mc_fwd, mc_paths = run_monte_carlo(df, horizon, freq, n_sims, scenario)
                if mc_fwd is not None:
                    forecasts["Monte Carlo"] = mc_fwd

        if not forecasts:
            st.error("No models ran successfully. Please install the required packages.")
            return

        st.markdown('<div class="section-header">Forecast Chart</div>', unsafe_allow_html=True)
        fig_fc = chart_forecast(df, forecasts, f"Financial Forecast — {scenario.upper()} Scenario")
        st.plotly_chart(fig_fc, use_container_width=True)

        # Summary KPIs per model
        st.markdown('<div class="section-header">Forecast Summary</div>', unsafe_allow_html=True)
        summary_cols = st.columns(len(forecasts))
        for i, (name, fwd) in enumerate(forecasts.items()):
            terminal = fwd["yhat"].iloc[-1]
            change   = (terminal - series.iloc[-1]) / series.iloc[-1] * 100
            c_col = {"Prophet": COLORS["prophet"], "ARIMA": COLORS["arima"],
                     "XGBoost": COLORS["xgb"],    "Monte Carlo": COLORS["monte"]}.get(name, "#fff")
            summary_cols[i].markdown(f"""
            <div class="metric-card" style="border-top-color:{c_col};">
              <div class="metric-label">{name} — {horizon}-step</div>
              <div class="metric-value" style="font-size:1.35rem;">{terminal:,.0f}</div>
              <div class="{'metric-delta-pos' if change>0 else 'metric-delta-neg'}">
                {'▲' if change>0 else '▼'} {abs(change):.1f}% vs last
              </div>
            </div>""", unsafe_allow_html=True)

        # Forecast table
        st.markdown('<div class="section-header">Forecast Table</div>', unsafe_allow_html=True)
        table_rows = []
        all_dates = forecasts[list(forecasts.keys())[0]]["ds"]
        for _, d in enumerate(all_dates):
            row = {"Date": d.strftime("%Y-%m-%d")}
            for name, fwd in forecasts.items():
                match = fwd[fwd["ds"] == d]
                if not match.empty:
                    row[f"{name} (Forecast)"] = round(match["yhat"].values[0], 2)
                    row[f"{name} (Lower)"]    = round(match["lower"].values[0], 2)
                    row[f"{name} (Upper)"]    = round(match["upper"].values[0], 2)
            table_rows.append(row)
        table_df = pd.DataFrame(table_rows)
        st.dataframe(table_df, hide_index=True, use_container_width=True)
        st.session_state["forecast_table"] = table_df

    # ══ TAB 3 — Monte Carlo ═══════════════════════════════════════════════════
    with tab3:
        if not run_btn:
            st.markdown('<div style="padding:3rem;text-align:center;color:#44445a;">Run the forecast first.</div>',
                        unsafe_allow_html=True)
            return

        if not run_mc_m or mc_paths is None:
            st.info("Enable Monte Carlo in the sidebar to see this view.")
            return

        fwd_dates = mc_fwd["ds"]
        st.markdown('<div class="section-header">Simulation Fan Chart</div>', unsafe_allow_html=True)
        fig_mc = chart_monte_carlo_fan(df, mc_paths, fwd_dates, n_show=min(200, n_sims))
        st.plotly_chart(fig_mc, use_container_width=True)

        col_mc1, col_mc2 = st.columns(2)
        with col_mc1:
            st.markdown('<div class="section-header">Outcome Distribution (Terminal)</div>',
                        unsafe_allow_html=True)
            terminal_vals = mc_paths[:, -1]
            fig_term = go.Figure()
            fig_term.add_trace(go.Histogram(
                x=terminal_vals, nbinsx=60,
                marker_color=COLORS["monte"],
                marker_line=dict(width=0.3, color="#09090f"),
            ))
            for p, col, label in [(10, COLORS["arima"], "P10"), (50, "#fff", "Median"), (90, COLORS["red"], "P90")]:
                fig_term.add_vline(x=np.percentile(terminal_vals, p),
                                   line_dash="dash", line_color=col, line_width=1.5,
                                   annotation_text=label,
                                   annotation_font_color=col,
                                   annotation_position="top")
            fig_term.update_layout(**CHART_LAYOUT, showlegend=False,
                                   title=dict(text="Terminal Value Distribution",
                                              font=dict(color="#e8e8f0", size=13), x=0.01))
            st.plotly_chart(fig_term, use_container_width=True)

        with col_mc2:
            st.markdown('<div class="section-header">Risk Summary Table</div>', unsafe_allow_html=True)
            pctls = [5, 10, 25, 50, 75, 90, 95]
            risk_data = pd.DataFrame({
                "Percentile": [f"P{p}" for p in pctls],
                "Terminal Value": [f"{np.percentile(terminal_vals, p):,.0f}" for p in pctls],
                "vs Last Actual": [f"{(np.percentile(terminal_vals, p)/series.iloc[-1]-1)*100:+.1f}%" for p in pctls],
            })
            st.dataframe(risk_data, hide_index=True, use_container_width=True)

            var_95 = np.percentile(terminal_vals, 5)
            prob_loss = np.mean(terminal_vals < series.iloc[-1]) * 100
            st.markdown(f"""
            <div class="insight-box">
              🎲 <b>Risk Metrics</b><br>
              <b>VaR (95%)</b>: {var_95:,.0f} &nbsp;|&nbsp;
              <b>Prob. of Loss</b>: {prob_loss:.1f}% &nbsp;|&nbsp;
              <b>Scenario</b>: {scenario.upper()}<br>
              Based on {n_sims:,} simulated paths over {horizon} steps.
            </div>""", unsafe_allow_html=True)

    # ══ TAB 4 — Model Comparison ══════════════════════════════════════════════
    with tab4:
        if not run_btn or not insample:
            st.markdown('<div style="padding:3rem;text-align:center;color:#44445a;">Run the forecast first.</div>',
                        unsafe_allow_html=True)
            return

        scores = {}
        for name, ins_df in insample.items():
            merged = df.merge(ins_df, on="ds", how="inner")
            if merged.empty: continue
            actual_vals = merged["y"].values
            pred_vals   = merged["y_pred"].values
            scores[name] = {
                "MAPE":  round(mape(actual_vals, pred_vals), 3),
                "SMAPE": round(smape(actual_vals, pred_vals), 3),
                "RMSE":  round(rmse(actual_vals, pred_vals), 2),
            }

        if scores:
            best_model = min(scores, key=lambda m: scores[m]["MAPE"])
            st.markdown('<div class="section-header">Model Accuracy Scores (In-Sample)</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart_model_errors(scores), use_container_width=True)

            # Table
            rows_html = ""
            for name, s in scores.items():
                is_best = name == best_model
                best_tag = '<span class="best-label">BEST</span>' if is_best else ""
                row_cls = 'class="best-model"' if is_best else ""
                rows_html += f"""<tr {row_cls}>
                  <td>{name}{best_tag}</td>
                  <td>{s['MAPE']:.3f}%</td>
                  <td>{s['SMAPE']:.3f}%</td>
                  <td>{s['RMSE']:,.2f}</td>
                </tr>"""

            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:1rem;">
              <table class="model-table">
                <thead><tr>
                  <th>Model</th><th>MAPE (%)</th><th>sMAPE (%)</th><th>RMSE</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-box" style="margin-top:1rem;">
              🏆 <b>Recommendation:</b> <b>{best_model}</b> achieved the lowest MAPE of
              <b>{scores[best_model]['MAPE']:.2f}%</b> on in-sample data.
              For production deployment, validate on a held-out test window and ensemble
              the top two models to reduce forecast variance.
            </div>""", unsafe_allow_html=True)

            # In-sample fit plot
            st.markdown('<div class="section-header">In-Sample Fit Overlay</div>', unsafe_allow_html=True)
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(
                x=df["ds"], y=df["y"], mode="lines",
                line=dict(color=COLORS["actual"], width=2),
                name="Actual",
            ))
            color_map = {"Prophet": COLORS["prophet"], "ARIMA": COLORS["arima"],
                         "XGBoost": COLORS["xgb"]}
            for name, ins_df in insample.items():
                fig_fit.add_trace(go.Scatter(
                    x=ins_df["ds"], y=ins_df["y_pred"], mode="lines",
                    line=dict(color=color_map.get(name, "#fff"), width=1.5, dash="dot"),
                    name=f"{name} Fit",
                ))
            fig_fit.update_layout(**CHART_LAYOUT,
                                  title=dict(text="In-Sample Fit vs Actual",
                                             font=dict(color="#e8e8f0", size=14), x=0.01))
            st.plotly_chart(fig_fit, use_container_width=True)

        st.session_state["scores"] = scores

    # ══ TAB 5 — Export ════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">Export Forecast Outputs</div>', unsafe_allow_html=True)

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.markdown('<div class="insight-box">📄 <b>Forecast Table (CSV)</b><br>Download the full forecast table with all models, confidence intervals, and dates.</div>',
                        unsafe_allow_html=True)
            if "forecast_table" in st.session_state:
                st.download_button(
                    "Download Forecast CSV",
                    data=to_csv_bytes(st.session_state["forecast_table"]),
                    file_name=f"fincast_forecast_{datetime.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Run the forecast first.")

        with ex_col2:
            st.markdown('<div class="insight-box">📊 <b>Historical Data (CSV)</b><br>Export the cleaned historical dataset used for modelling.</div>',
                        unsafe_allow_html=True)
            st.download_button(
                "Download Historical CSV",
                data=to_csv_bytes(df),
                file_name=f"fincast_historical_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        if "forecast_table" in st.session_state:
            st.markdown('<div class="section-header">Excel Export (All Sheets)</div>', unsafe_allow_html=True)
            sheets = {
                "Historical": df,
                "Forecast": st.session_state["forecast_table"],
            }
            if "scores" in st.session_state and st.session_state["scores"]:
                scores_df = pd.DataFrame(st.session_state["scores"]).T.reset_index()
                scores_df.columns = ["Model", "MAPE", "SMAPE", "RMSE"]
                sheets["Model Scores"] = scores_df
            st.download_button(
                "📥  Download Full Excel Report",
                data=to_excel_bytes(sheets),
                file_name=f"FinCast_Report_{datetime.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.markdown("""
            <div class="insight-box" style="margin-top:1rem;">
              💼 <b>Board-Ready Export:</b> The Excel file contains three sheets —
              Historical data, full forecast table with confidence intervals,
              and model accuracy scores. Ready to paste into any board deck or budget review.
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

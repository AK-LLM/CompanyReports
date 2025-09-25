import io
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Naked Fundamentals PRO (Streamlit)", layout="wide")


# ============================
# Utilities
# ============================

def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for to_excel (no tz, no lists/dicts, no inf)."""
    if df is None or df.empty:
        return df
    df = df.copy()
    # Drop timezone from datetime-like columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                if getattr(df[col].dt, "tz", None) is not None:
                    df[col] = df[col].dt.tz_localize(None)
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
        # Convert containers to string
        if df[col].apply(lambda x: isinstance(x, (list, dict, tuple, set))).any():
            df[col] = df[col].astype(str)
    # Replace infinities
    df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
    return df


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _pick_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """Find first matching row (case-insensitive contains) and return the row series with most-recent value first."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    idx = df.index.astype(str).str.lower()
    for name in candidates:
        m = idx.str.contains(name.lower(), regex=False)
        if m.any():
            s = df.loc[idx[m]].iloc[0]
            # reorder newest first for yfinance statements (columns are newest first already)
            return s.dropna()
    return None


def _latest_value(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    row = _pick_row(df, candidates)
    if row is None or len(row) == 0:
        return None
    return _safe_float(row.iloc[0])


# ============================
# Data Models
# ============================

@dataclass
class Core:
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    price: Optional[float] = None
    mktcap: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    shares_out: Optional[float] = None

    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None

    equity: Optional[float] = None
    debt: Optional[float] = None
    cash: Optional[float] = None
    current_assets: Optional[float] = None
    current_liab: Optional[float] = None
    goodwill: Optional[float] = None
    inventory: Optional[float] = None

    ev: Optional[float] = None
    pe: Optional[float] = None
    ps: Optional[float] = None
    pb: Optional[float] = None
    ev_ebitda: Optional[float] = None
    de: Optional[float] = None
    int_cov: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    gw_assets: Optional[float] = None
    fcf: Optional[float] = None
    fcf_yield: Optional[float] = None
    roic: Optional[float] = None
    ccc: Optional[float] = None  # cash conversion cycle (days)


# ============================
# Yahoo Fetch (fast & robust)
# ============================

def _with_retries(fn, attempts=3, delay=0.5):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(ticker: str, hist_period: str = "1y") -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (info, income_stmt, balance_sheet, cashflow, history)."""
    t = yf.Ticker(ticker)

    # Fast info first
    info = {}
    try:
        fi = getattr(t, "fast_info", {}) or {}
        if fi:
            info.update({
                "currentPrice": fi.get("last_price"),
                "marketCap": fi.get("market_cap"),
                "beta": fi.get("beta"),
                "dividendYield": fi.get("dividend_yield"),
                "sharesOutstanding": fi.get("shares")
            })
    except Exception:
        pass

    # Fallback to full info only if we still need fields
    def _full_info():
        i = t.get_info() if hasattr(t, "get_info") else t.info
        return i or {}

    if not info.get("currentPrice") or not info.get("marketCap"):
        try:
            i = _with_retries(_full_info)
            for k in ["currentPrice", "marketCap", "beta", "dividendYield", "sector", "industry", "shortName",
                      "longName", "sharesOutstanding"]:
                if k in i and info.get(k) is None:
                    info[k] = i.get(k)
        except Exception:
            pass

    # Financials (Yahoo can be flaky; wrap with retries)
    def _get_attr(name):
        return getattr(t, name, pd.DataFrame())

    income = _with_retries(lambda: _get_attr("financials") or pd.DataFrame())
    balance = _with_retries(lambda: _get_attr("balance_sheet") or pd.DataFrame())
    cashflw = _with_retries(lambda: _get_attr("cashflow") or pd.DataFrame())

    # Price history
    hist = pd.DataFrame()
    try:
        hist = _with_retries(lambda: t.history(period=hist_period, auto_adjust=False) or pd.DataFrame())
    except Exception:
        pass

    return info, income, balance, cashflw, hist


# ============================
# Feature Calculations
# ============================

def build_core(ticker: str, info: dict, inc: pd.DataFrame, bal: pd.DataFrame, cfs: pd.DataFrame) -> Core:
    c = Core(ticker=ticker)

    # Basic info
    c.name = info.get("longName") or info.get("shortName") or ticker
    c.sector = info.get("sector")
    c.industry = info.get("industry")
    c.price = _safe_float(info.get("currentPrice"))
    c.mktcap = _safe_float(info.get("marketCap"))
    c.beta = _safe_float(info.get("beta"))
    divy = info.get("dividendYield")
    c.dividend_yield = _safe_float(divy if isinstance(divy, (int, float)) and divy < 1.0 else (divy/100 if divy else None))
    c.shares_out = _safe_float(info.get("sharesOutstanding"))

    # Income statement
    c.revenue = _latest_value(inc, ["total revenue", "revenue"])
    c.gross_profit = _latest_value(inc, ["gross profit"])
    c.ebit = _latest_value(inc, ["ebit", "operating income"])
    c.ebitda = _latest_value(inc, ["ebitda"])
    c.net_income = _latest_value(inc, ["net income", "net income applicable to common shares"])

    # Balance sheet
    c.equity = _latest_value(bal, ["total stockholder", "total shareholders'", "total equity"])
    c.debt = _latest_value(bal, ["total debt", "long term debt", "long-term debt"])
    c.cash = _latest_value(bal, ["cash", "cash and cash equivalents", "cash and short term investments"])
    c.current_assets = _latest_value(bal, ["total current assets"])
    c.current_liab = _latest_value(bal, ["total current liabilities"])
    c.goodwill = _latest_value(bal, ["goodwill"])
    c.inventory = _latest_value(bal, ["inventory"])

    # Cashflow
    c.fcf = _latest_value(cfs, ["free cash flow", "free cash flow to the firm"])

    # Ratios
    if c.mktcap and c.price and c.shares_out:

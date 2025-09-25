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
                "sharesOutstanding": fi.get("shares"),
            })
    except Exception:
        pass

    # Fallback to full info only if still missing fields
    def _full_info():
        i = t.get_info() if hasattr(t, "get_info") else t.info
        return i or {}

    if not info.get("currentPrice") or not info.get("marketCap") or not info.get("sector") or not info.get("industry"):
        try:
            i = _with_retries(_full_info)
            for k in ["currentPrice", "marketCap", "beta", "dividendYield", "sector", "industry",
                      "shortName", "longName", "sharesOutstanding"]:
                if k in i and info.get(k) is None:
                    info[k] = i.get(k)
        except Exception:
            pass

    # Financials (wrap with retries)
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
    # P/S and P/B depend only on market cap and fundamentals (not price/shares directly)
    if c.mktcap and c.revenue:
        c.ps = c.mktcap / c.revenue
    if c.mktcap and c.equity:
        c.pb = c.mktcap / c.equity

    # P/E via EPS
    if c.price and c.shares_out and c.net_income:
        eps = c.net_income / c.shares_out if c.shares_out else None
        c.pe = (c.price / eps) if (eps and eps != 0) else None

    # EV and EV/EBITDA
    if c.mktcap is not None:
        net_debt = (c.debt or 0) - (c.cash or 0)
        c.ev = c.mktcap + net_debt
    if c.ev and c.ebitda:
        c.ev_ebitda = c.ev / c.ebitda

    # Leverage & Liquidity
    if c.debt is not None and c.equity:
        c.de = c.debt / c.equity if c.equity else None
    if c.current_assets and c.current_liab:
        c.current_ratio = c.current_assets / c.current_liab if c.current_liab else None
        inv = c.inventory or 0
        c.quick_ratio = (c.current_assets - inv) / c.current_liab if c.current_liab else None
    if c.goodwill:
        assets = _latest_value(bal, ["total assets"])
        c.gw_assets = c.goodwill / assets if assets else None

    # Interest coverage (approx, EBIT / interest expense)
    int_exp = _latest_value(inc, ["interest expense"])
    if c.ebit and int_exp:
        c.int_cov = c.ebit / abs(int_exp) if int_exp else None

    # ROIC (approx): NOPAT / (Equity + Debt - Cash)
    nopat = (c.ebit * 0.79) if c.ebit else None  # ~21% tax proxy
    invested_cap = None
    if c.equity is not None or c.debt is not None:
        invested_cap = (c.equity or 0) + (c.debt or 0) - (c.cash or 0)
    if nopat and invested_cap:
        c.roic = nopat / invested_cap if invested_cap else None

    # FCF yield
    if c.fcf and c.mktcap:
        c.fcf_yield = c.fcf / c.mktcap

    # Cash Conversion Cycle (rough)
    ar = _latest_value(bal, ["accounts receivable"])
    ap = _latest_value(bal, ["accounts payable"])
    cogs = _latest_value(inc, ["cost of revenue", "cost of goods sold"])
    if c.revenue and cogs and ar and ap and c.inventory:
        AR_days = (ar / (c.revenue / 365.0)) if c.revenue else None
        INV_days = (c.inventory / (cogs / 365.0)) if cogs else None
        AP_days = (ap / (cogs / 365.0)) if cogs else None
        if AR_days and INV_days and AP_days:
            c.ccc = AR_days + INV_days - AP_days

    return c


def momentum_and_prices(hist: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    out: Dict[str, float] = {}
    if not isinstance(hist, pd.DataFrame) or hist.empty:
        return out, pd.DataFrame()
    close = hist["Close"].dropna()
    if close.empty:
        return out, pd.DataFrame()

    # returns
    out["ret_1m"] = float(close.pct_change(21).iloc[-1]) if len(close) >= 22 else None
    out["ret_3m"] = float(close.pct_change(63).iloc[-1]) if len(close) >= 64 else None
    out["ret_6m"] = float(close.pct_change(126).iloc[-1]) if len(close) >= 127 else None
    out["ret_12m"] = float(close.pct_change(252).iloc[-1]) if len(close) >= 253 else None

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["sma50"] = float(sma50.iloc[-1]) if not sma50.dropna().empty else None
    out["sma200"] = float(sma200.iloc[-1]) if not sma200.dropna().empty else None
    out["price"] = float(close.iloc[-1])
    out["hi_52w"] = float(close.tail(252).max()) if len(close) >= 252 else None
    out["px_vs_52w_hi"] = (out["price"] / out["hi_52w"] - 1.0) if (out.get("price") and out.get("hi_52w")) else None

    df = pd.DataFrame({"Date": close.index, "Close": close.values, "SMA50": sma50.values, "SMA200": sma200.values})
    return out, df


def simple_score(core: Core, mom: Dict[str, float]) -> Tuple[float, Dict[str, float], str]:
    """Return (score/100, parts, verdict)."""
    parts: Dict[str, float] = {}

    # Value: low multiples, high FCF yield
    v = 0.0
    if core.pe and core.pe > 0:
        v += max(0, min(30, (30 - core.pe))) / 30 * 30
    if core.ev_ebitda and core.ev_ebitda > 0:
        v += max(0, min(20, (20 - core.ev_ebitda))) / 20 * 20
    if core.fcf_yield:
        v += max(0, min(25, core.fcf_yield * 100))
    parts["Value"] = v

    # Quality: ROIC, leverage, interest coverage
    q = 0.0
    if core.roic:
        q += max(0, min(25, core.roic * 100))
    if core.de is not None:
        q += max(0, min(10, (2.0 - core.de) * 5)) if core.de <= 2 else 0
    if core.int_cov:
        q += max(0, min(10, (core.int_cov / 10) * 10))
    parts["Quality"] = q

    # Momentum: returns + SMA cross
    m = 0.0
    for key, weight in [("ret_12m", 10), ("ret_6m", 10), ("ret_3m", 5)]:
        if mom.get(key) is not None:
            m += max(0, min(weight, mom[key] * 100))
    if mom.get("sma50") and mom.get("sma200"):
        m += 10 if mom["sma50"] >= mom["sma200"] else 0
    parts["Momentum"] = m

    score = sum(parts.values())
    verdict = "BUY" if score >= 65 else ("HOLD" if score >= 45 else "SELL")
    return score, parts, verdict


# ============================
# Peer Medians
# ============================

@st.cache_data(ttl=3600, show_spinner=False)
def peer_medians(peers: List[str]) -> Dict[str, float]:
    meds: Dict[str, float] = {}
    if not peers:
        return meds

    def _grab(tk: str):
        info, inc, bal, cfs, _ = fetch_all(tk, "1y")
        c = build_core(tk, info, inc, bal, cfs)
        return {
            "P/E": c.pe, "EV/EBITDA": c.ev_ebitda, "P/S": c.ps, "P/B": c.pb,
            "Debt/Equity": c.de, "FCF Yield": c.fcf_yield, "ROIC": c.roic
        }

    rows = []
    with ThreadPoolExecutor(max_workers=min(5, len(peers))) as ex:
        futs = [ex.submit(_grab, p) for p in peers]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception:
                pass

    if not rows:
        return meds
    df = pd.DataFrame(rows)
    meds = df.median(numeric_only=True).to_dict()
    return meds


# ============================
# DCF (simple)
# ============================

def simple_dcf(core: Core, rf: float, mkt_prem: float, term_growth: float, years: int = 5) -> Tuple[pd.DataFrame, float]:
    """Very simple FCFF model → Equity value per share (approx)."""
    # WACC via CAPM for Re; Rd ~ rf + 1.5% spread; weights by market
    beta = core.beta or 1.0
    re = rf + beta * mkt_prem
    rd = rf + 0.015
    E = core.mktcap or 0
    D = core.debt or 0
    if E + D == 0:
        wacc = re
    else:
        Tc = 0.21
        wacc = (E/(E+D))*re + (D/(E+D))*rd*(1-Tc)

    # Base FCF
    fcf0 = core.fcf if core.fcf else (core.net_income or 0)
    if not fcf0 or fcf0 <= 0:
        fcf0 = (core.ebit or 0) * 0.79  # rough NOPAT proxy

    # Assume modest decaying growth to terminal
    g1 = min(0.12, max(0.00, mkt_prem))
    gs = np.linspace(g1, term_growth, years)
    fcf_list = []
    pv = 0.0
    for t, g in enumerate(gs, start=1):
        fcf_t = fcf0 * ((1+g) ** t)
        fcf_list.append(fcf_t)
        pv += fcf_t / ((1 + wacc) ** t)
    # Terminal value (Gordon)
    tv = fcf_list[-1] * (1 + term_growth) / (wacc - term_growth) if wacc > term_growth else 0.0
    pv += tv / ((1 + wacc) ** years)

    # Equity value
    net_debt = (core.debt or 0) - (core.cash or 0)
    eq_val = pv - net_debt
    per_share = eq_val / core.shares_out if core.shares_out else None

    df = pd.DataFrame({
        "Year": list(range(1, years+1)),
        "Growth": [f"{g:.2%}" for g in gs],
        "FCF": fcf_list,
        "DiscountFactor": [1/((1+wacc)**t) for t in range(1, years+1)]
    })
    return df, per_share or 0.0


def dcf_sensitivity(core: Core, rf: float, prem_list: List[float], g_list: List[float]) -> List[Dict]:
    rows = []
    for prem in prem_list:
        for g in g_list:
            _, ps = simple_dcf(core, rf, prem, g, years=5)
            rows.append({"MktPrem": prem, "TerminalG": g, "Price/Share": ps})
    return rows


# ============================
# One-ticker analysis
# ============================

def analyze_one(ticker: str, hist_period: str, rf: float, mkt_prem: float, term_g: float):
    info, inc, bal, cfs, hist = fetch_all(ticker, hist_period)
    core = build_core(ticker, info, inc, bal, cfs)
    mom, price_df = momentum_and_prices(hist)
    score, parts, verdict = simple_score(core, mom)

    dcf_table, dcf_px = simple_dcf(core, rf, mkt_prem, term_g, years=5)
    cap = {
        "Market Cap": core.mktcap,
        "Enterprise Value": core.ev,
        "Debt": core.debt,
        "Cash": core.cash,
        "Net Debt": (core.debt or 0) - (core.cash or 0),
        "Shares Out": core.shares_out
    }
    return core, dcf_table, mom, price_df, score, parts, verdict, cap, dcf_px


def _fmt(v, pct=False):
    if v is None:
        return "—"
    try:
        if pct:
            return f"{v*100:.2f}%"
        return f"{v:,.2f}"
    except Exception:
        return str(v)


# ============================
# UI
# ============================

st.title("🧠 Naked Fundamentals PRO — Streamlit")
st.caption("Fundamentals • Value/Quality/Momentum • Peer medians • DCF • Export")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Single Ticker", "Compare 3"], index=0)
    hist_period = st.selectbox("Price History Period", ["1y", "3y", "5y"], index=0)
    if mode == "Single Ticker":
        tk1 = st.text_input("Ticker", "AAPL").upper().strip()
    else:
        tk1 = st.text_input("Ticker 1", "AAPL").upper().strip()
        tk2 = st.text_input("Ticker 2", "MSFT").upper().strip()
        tk3 = st.text_input("Ticker 3", "GOOGL").upper().strip()

    st.markdown("---")
    st.subheader("Valuation Settings")
    rf = st.number_input("Risk-free rate (rf)", value=0.045, step=0.005, format="%.3f")
    mkt_prem = st.number_input("Market risk premium", value=0.055, step=0.005, format="%.3f")
    term_g = st.number_input("Terminal growth", value=0.020, step=0.002, format="%.3f")

    st.markdown("---")
    peer_str = st.text_input("Peers (comma-separated, optional)", "")
    st.caption("Tip: add peers to compute medians for P/E, EV/EBITDA, FCF Yield, ROIC, etc.")

    # versions
    try:
        import sys
        st.caption(f"Py {sys.version.split()[0]} • pandas {pd.__version__} • numpy {np.__version__} • yfinance {getattr(yf, '__version__','?')}")
    except Exception:
        pass


if mode == "Single Ticker":
    if not tk1:
        st.warning("Enter a ticker.")
        st.stop()

    with st.spinner(f"Fetching {tk1}..."):
        core, dcf_table, mom, price_df, score, parts, verdict, cap, dcf_px = analyze_one(tk1, hist_period, rf, mkt_prem, term_g)

    # Header
    st.subheader(f"{tk1} — {core.name or ''}")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Price", _fmt(core.price))
    colB.metric("Market Cap", _fmt(core.mktcap))
    colC.metric("Beta", _fmt(core.beta))
    colD.metric("Dividend Yield", _fmt(core.dividend_yield, pct=True))

    # Score
    st.markdown("### Score & Verdict")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{score:.1f}/100")
    c2.metric("Value", f"{parts.get('Value',0):.1f}")
    c3.metric("Quality", f"{parts.get('Quality',0):.1f}")
    c4.metric("Momentum", f"{parts.get('Momentum',0):.1f}")
    st.success(f"**{verdict}** — simple composite of V/Q/M.")

    # Price chart
    if not price_df.empty:
        st.markdown("### Price & Moving Averages")
        fig = px.line(price_df, x="Date", y=["Close", "SMA50", "SMA200"])
        st.plotly_chart(fig, use_container_width=True)

    # Fundamentals table
    st.markdown("### Key Fundamentals")
    rows = [
        ("Sector", core.sector),
        ("Industry", core.industry),
        ("P/E", core.pe),
        ("EV/EBITDA", core.ev_ebitda),
        ("P/S", core.ps),
        ("P/B", core.pb),
        ("Debt/Equity", core.de),
        ("Interest Coverage", core.int_cov),
        ("Current Ratio", core.current_ratio),
        ("Quick Ratio", core.quick_ratio),
        ("FCF", core.fcf),
        ("FCF Yield", core.fcf_yield),
        ("ROIC", core.roic),
        ("Cash Conversion Cycle (days)", core.ccc),
    ]
    df_fund = pd.DataFrame({"Metric": [r[0] for r in rows],
                            "Value": [_fmt(r[1], pct=("Yield" in r[0] or r[0]=="ROIC")) for r in rows]})
    st.dataframe(df_fund, use_container_width=True)

    # Capital Allocation
    st.markdown("### Capital Allocation")
    cap_df = pd.DataFrame({"Metric": list(cap.keys()), "Value": list(cap.values())})
    st.dataframe(cap_df, use_container_width=True)

    # Peer Medians
    if peer_str.strip():
        st.markdown("### Peer Medians")
        peers = sorted({p.strip().upper() for p in peer_str.split(",") if p.strip()})
        with st.spinner("Fetching peers..."):
            meds = peer_medians(peers)
        if meds:
            bench = pd.DataFrame([
                {"Metric": "P/E", "Company": core.pe, "Peers (Median)": meds.get("P/E")},
                {"Metric": "EV/EBITDA", "Company": core.ev_ebitda, "Peers (Median)": meds.get("EV/EBITDA")},
                {"Metric": "P/S", "Company": core.ps, "Peers (Median)": meds.get("P/S")},
                {"Metric": "P/B", "Company": core.pb, "Peers (Median)": meds.get("P/B")},
                {"Metric": "Debt/Equity", "Company": core.de, "Peers (Median)": meds.get("Debt/Equity")},
                {"Metric": "FCF Yield", "Company": core.fcf_yield, "Peers (Median)": meds.get("FCF Yield")},
                {"Metric": "ROIC", "Company": core.roic, "Peers (Median)": meds.get("ROIC")},
            ])
            st.dataframe(bench, use_container_width=True)
        else:
            st.info("No peer data available (Yahoo may be rate-limiting).")

    # DCF
    st.markdown("### DCF (simple)")
    st.dataframe(dcf_table, use_container_width=True)
    if dcf_px:
        st.info(f"**DCF Fair Value (approx)**: {_fmt(dcf_px)} per share")

    # Sensitivity
    rows = dcf_sensitivity(core, rf, prem_list=[0.045, 0.055, 0.065], g_list=[0.015, 0.02, 0.025])
    sens_df = pd.DataFrame(rows)
    st.markdown("**DCF Sensitivity (Market Premium × Terminal Growth)**")
    st.dataframe(sens_df, use_container_width=True)

    # Export
    st.markdown("### Export")
    if st.button("Build Excel Workbook"):
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            sanitize_for_excel(df_fund).to_excel(writer, index=False, sheet_name="Summary")
            sanitize_for_excel(cap_df).to_excel(writer, index=False, sheet_name="CapitalAlloc")
            sanitize_for_excel(dcf_table).to_excel(writer, index=False, sheet_name="DCF")
            sanitize_for_excel(sens_df).to_excel(writer, index=False, sheet_name="Sensitivity")
            if not price_df.empty:
                pdf = price_df.copy()
                if "Date" in pdf.columns and pd.api.types.is_datetime64_any_dtype(pdf["Date"]):
                    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce").dt.tz_localize(None)
                sanitize_for_excel(pdf).to_excel(writer, index=False, sheet_name="Prices")
        st.download_button(
            "Download Excel",
            bio.getvalue(),
            file_name=f"{tk1}_Naked_PRO.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    # Compare 3
    tickers = [t for t in [tk1, tk2, tk3] if t]
    if not tickers:
        st.warning("Enter at least one ticker.")
        st.stop()

    st.subheader("Compare 3 — Summary")
    results = []
    charts = []

    def _job(tk):
        try:
            return tk, analyze_one(tk, hist_period, rf, mkt_prem, term_g)
        except Exception as e:
            return tk, e

    with st.spinner("Fetching tickers in parallel..."):
        with ThreadPoolExecutor(max_workers=min(3, len(tickers))) as ex:
            futs = [ex.submit(_job, tk) for tk in tickers]
            for f in as_completed(futs):
                tk, res = f.result()
                if isinstance(res, Exception):
                    st.error(f"{tk}: {res}")
                    continue
                core, dcf_table, mom, price_df, score, parts, verdict, cap, dcf_px = res
                results.append({
                    "Ticker": tk,
                    "Name": core.name,
                    "Price": core.price,
                    "Score": round(score, 1),
                    "Verdict": verdict,
                    "P/E": core.pe,
                    "EV/EBITDA": core.ev_ebitda,
                    "D/E": core.de,
                    "FCF Yield": core.fcf_yield,
                    "ROIC": core.roic
                })
                if not price_df.empty:
                    charts.append((tk, price_df))

    if results:
        cmp_df = pd.DataFrame(results)
        st.dataframe(cmp_df, use_container_width=True)

        # Normalized price chart (last=100)
        if charts:
            st.markdown("### Normalized Prices (last = 100)")
            parts = []
            for tk, dfp in charts:
                ser = dfp["Close"].dropna()
                if ser.empty:
                    continue
                base = ser.iloc[-1]
                parts.append(pd.DataFrame({"Date": dfp["Date"], "Ticker": tk, "Index": ser / base * 100}))
            if parts:
                chart_df = pd.concat(parts)
                fig = px.line(chart_df, x="Date", y="Index", color="Ticker")
                st.plotly_chart(fig, use_container_width=True)

        # Download comparison
        st.download_button(
            "Download Comparison CSV",
            cmp_df.to_csv(index=False).encode("utf-8"),
            file_name="Compare3_summary.csv",
            mime="text/csv"
        )

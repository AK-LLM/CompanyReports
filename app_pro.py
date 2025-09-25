# app_pro.py — External Signals + Altman Z / Beneish M-Score (proper variants & thresholds)
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

st.set_page_config(page_title="Naked Fundamentals PRO", layout="wide")

# ==============
# Helpers
# ==============

def _num_suffix(x: Optional[float], pct: bool=False) -> str:
    if x is None:
        return "—"
    try:
        if pct:
            return f"{x*100:.2f}%"
        abx = abs(x)
        if abx >= 1e12: return f"{x/1e12:.2f}T"
        if abx >= 1e9:  return f"{x/1e9:.2f}B"
        if abx >= 1e6:  return f"{x/1e6:.2f}M"
        if abx >= 1e3:  return f"{x/1e3:.2f}K"
        return f"{x:.2f}"
    except Exception:
        return str(x)

def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                if getattr(df[col].dt, "tz", None) is not None:
                    df[col] = df[col].dt.tz_localize(None)
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
        if df[col].apply(lambda v: isinstance(v,(list,dict,tuple,set))).any():
            df[col] = df[col].astype(str)
    df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
    return df

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        f = float(x)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except Exception: return None

def _pick_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    if not isinstance(df, pd.DataFrame) or df.empty: return None
    idx_lower = pd.Index(df.index.astype(str).str.lower())
    for name in candidates:
        mask = idx_lower.str.contains(name.lower(), regex=False)
        if mask.any():
            s = df.loc[mask].iloc[0]
            return s.dropna()
    return None

def _series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    row = _pick_row(df, candidates)
    return row if (row is not None and len(row)>0) else None

def _latest_value(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    row = _series(df, candidates)
    if row is None: return None
    return _safe_float(row.iloc[0])

def _latest_prev(df: pd.DataFrame, candidates: List[str]) -> Tuple[Optional[float], Optional[float]]:
    row = _series(df, candidates)
    if row is None or len(row)<2: return None, None
    return _safe_float(row.iloc[0]), _safe_float(row.iloc[1])

# ==============
# Data model
# ==============

@dataclass
class Core:
    ticker: str
    name: Optional[str]=None; sector: Optional[str]=None; industry: Optional[str]=None
    price: Optional[float]=None; mktcap: Optional[float]=None; beta: Optional[float]=None
    dividend_yield: Optional[float]=None; shares_out: Optional[float]=None
    revenue: Optional[float]=None; gross_profit: Optional[float]=None; ebit: Optional[float]=None
    ebitda: Optional[float]=None; net_income: Optional[float]=None
    equity: Optional[float]=None; debt: Optional[float]=None; cash: Optional[float]=None
    current_assets: Optional[float]=None; current_liab: Optional[float]=None
    goodwill: Optional[float]=None; inventory: Optional[float]=None
    ev: Optional[float]=None; pe: Optional[float]=None; ps: Optional[float]=None; pb: Optional[float]=None
    ev_ebitda: Optional[float]=None; de: Optional[float]=None; int_cov: Optional[float]=None
    current_ratio: Optional[float]=None; quick_ratio: Optional[float]=None; gw_assets: Optional[float]=None
    fcf: Optional[float]=None; fcf_yield: Optional[float]=None; roic: Optional[float]=None; ccc: Optional[float]=None

# ==============
# Fetch
# ==============

def _with_retries(fn, attempts=3, delay=0.5):
    for i in range(attempts):
        try: return fn()
        except Exception:
            if i==attempts-1: raise
            time.sleep(delay)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(ticker: str, hist_period: str="1y"):
    t = yf.Ticker(ticker)
    info = {}
    try:
        fi = getattr(t,"fast_info",{}) or {}
        if fi:
            info.update({
                "currentPrice": fi.get("last_price"),
                "marketCap": fi.get("market_cap"),
                "beta": fi.get("beta"),
                "dividendYield": fi.get("dividend_yield"),
                "sharesOutstanding": fi.get("shares"),
            })
    except Exception: pass
    def _full_info():
        i = t.get_info() if hasattr(t,"get_info") else t.info
        return i or {}
    if not info.get("currentPrice") or not info.get("marketCap") or not info.get("sector") or not info.get("industry"):
        try:
            i = _with_retries(_full_info)
            for k in ["currentPrice","marketCap","beta","dividendYield","sector","industry","shortName","longName","sharesOutstanding",
                      "targetMeanPrice","targetLowPrice","targetHighPrice","numberOfAnalystOpinions","recommendationMean","recommendationKey",
                      "sharesShort","shortPercentOfFloat","shortRatio"]:
                if k in i and info.get(k) is None: info[k]=i.get(k)
        except Exception: pass
    def _get_attr(name):
        try:
            df = getattr(t,name)
            return df if isinstance(df,pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    income  = _with_retries(lambda: _get_attr("financials"))
    balance = _with_retries(lambda: _get_attr("balance_sheet"))
    cashflw = _with_retries(lambda: _get_attr("cashflow"))
    def _hist():
        try:
            df = t.history(period=hist_period, auto_adjust=False)
            return df if isinstance(df,pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    hist = _with_retries(_hist)
    return info, income, balance, cashflw, hist

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history_only(ticker: str, hist_period: str="1y"):
    try:
        df = yf.Ticker(ticker).history(period=hist_period, auto_adjust=False)
        return df if isinstance(df,pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ==============
# Features (core)
# ==============

def build_core(ticker: str, info: dict, inc: pd.DataFrame, bal: pd.DataFrame, cfs: pd.DataFrame) -> Core:
    c = Core(ticker=ticker)
    c.name = info.get("longName") or info.get("shortName") or ticker
    c.sector = info.get("sector"); c.industry = info.get("industry")
    c.price = _safe_float(info.get("currentPrice")); c.mktcap = _safe_float(info.get("marketCap"))
    c.beta = _safe_float(info.get("beta"))
    divy = info.get("dividendYield")
    c.dividend_yield = _safe_float(divy if isinstance(divy,(int,float)) and divy<1.0 else (divy/100 if divy else None))
    c.shares_out = _safe_float(info.get("sharesOutstanding"))

    c.revenue = _latest_value(inc, ["total revenue","revenue"])
    c.gross_profit = _latest_value(inc, ["gross profit"])
    c.ebit = _latest_value(inc, ["ebit","operating income"])
    c.ebitda = _latest_value(inc, ["ebitda"])
    c.net_income = _latest_value(inc, ["net income","net income applicable to common shares"])

    c.equity = _latest_value(bal, ["total stockholder","total shareholders'","total equity"])
    c.debt = _latest_value(bal, ["total debt","long term debt","long-term debt"])
    c.cash = _latest_value(bal, ["cash","cash and cash equivalents","cash and short term investments"])
    c.current_assets = _latest_value(bal, ["total current assets"])
    c.current_liab = _latest_value(bal, ["total current liabilities"])
    c.goodwill = _latest_value(bal, ["goodwill"])
    c.inventory = _latest_value(bal, ["inventory"])

    c.fcf = _latest_value(cfs, ["free cash flow","free cash flow to the firm"])

    if c.mktcap and c.revenue: c.ps = c.mktcap / c.revenue
    if c.mktcap and c.equity:  c.pb = c.mktcap / c.equity

    if c.price and c.shares_out and c.net_income:
        eps = c.net_income / c.shares_out if c.shares_out else None
        c.pe = (c.price / eps) if (eps and eps != 0) else None

    if c.mktcap is not None:
        net_debt = (c.debt or 0) - (c.cash or 0)
        c.ev = c.mktcap + net_debt
    if c.ev and c.ebitda: c.ev_ebitda = c.ev / c.ebitda

    if c.debt is not None and c.equity:
        c.de = c.debt / c.equity if c.equity else None
    if c.current_assets and c.current_liab:
        c.current_ratio = c.current_assets / c.current_liab if c.current_liab else None
        inv = c.inventory or 0
        c.quick_ratio = (c.current_assets - inv) / c.current_liab if c.current_liab else None
    if c.goodwill:
        assets = _latest_value(bal, ["total assets"])
        c.gw_assets = c.goodwill / assets if assets else None

    int_exp = _latest_value(inc, ["interest expense"])
    if c.ebit and int_exp: c.int_cov = c.ebit / abs(int_exp) if int_exp else None

    # ROIC with average invested capital (Equity + Debt − Cash)
    assets_t, assets_p = _latest_prev(bal, ["total assets"])
    equity_t, equity_p = _latest_prev(bal, ["total stockholder","total shareholders'","total equity"])
    debt_t, debt_p = _latest_prev(bal, ["total debt","long term debt","long-term debt"])
    cash_t, cash_p = _latest_prev(bal, ["cash","cash and cash equivalents","cash and short term investments"])
    if c.ebit and assets_t and (equity_t is not None) and (debt_t is not None) and (cash_t is not None):
        ic_t = (equity_t or 0) + (debt_t or 0) - (cash_t or 0)
        ic_p = None
        if (equity_p is not None) or (debt_p is not None) or (cash_p is not None):
            ic_p = (equity_p or 0) + (debt_p or 0) - (cash_p or 0)
        avg_ic = (ic_t + ic_p)/2 if ic_p is not None else ic_t
        nopat = c.ebit * 0.79
        if avg_ic and avg_ic>0: c.roic = nopat/avg_ic

    if c.fcf and c.mktcap: c.fcf_yield = c.fcf / c.mktcap

    # CCC
    ar = _latest_value(bal, ["accounts receivable"])
    ap = _latest_value(bal, ["accounts payable"])
    cogs = _latest_value(inc, ["cost of revenue","cost of goods sold"])
    if c.revenue and cogs and ar and ap and c.inventory:
        AR_days = ar / (c.revenue/365.0); INV_days = c.inventory/(cogs/365.0); AP_days = ap/(cogs/365.0)
        c.ccc = AR_days + INV_days - AP_days

    return c

def momentum_and_prices(hist: pd.DataFrame, bench_hist: pd.DataFrame=None):
    out: Dict[str,float] = {}
    if not isinstance(hist,pd.DataFrame) or hist.empty: return out, pd.DataFrame()
    close = hist["Close"].dropna()
    if close.empty: return out, pd.DataFrame()
    out["ret_1m"] = float(close.pct_change(21).iloc[-1]) if len(close)>=22 else None
    out["ret_3m"] = float(close.pct_change(63).iloc[-1]) if len(close)>=64 else None
    out["ret_6m"] = float(close.pct_change(126).iloc[-1]) if len(close)>=127 else None
    out["ret_12m"] = float(close.pct_change(252).iloc[-1]) if len(close)>=253 else None
    sma50 = close.rolling(50).mean(); sma200 = close.rolling(200).mean()
    out["sma50"] = float(sma50.iloc[-1]) if not sma50.dropna().empty else None
    out["sma200"] = float(sma200.iloc[-1]) if not sma200.dropna().empty else None
    out["price"] = float(close.iloc[-1])
    out["hi_52w"] = float(close.tail(252).max()) if len(close)>=252 else None
    out["px_vs_52w_hi"] = (out["price"]/out["hi_52w"] - 1.0) if (out.get("price") and out.get("hi_52w")) else None

    daily = close.pct_change().dropna()
    out["vol_1y"] = float(daily.std()*np.sqrt(252)) if not daily.empty else None
    roll_max = close.cummax(); dd = (close/roll_max - 1.0).min() if not close.empty else None
    out["max_dd_1y"] = float(dd) if dd is not None else None

    if isinstance(bench_hist,pd.DataFrame) and not bench_hist.empty and "Close" in bench_hist:
        b = bench_hist["Close"].reindex(close.index).ffill().dropna()
        if not b.empty and len(b)==len(close):
            rs = (close/b)
            out["rs_6m"] = float(rs.iloc[-1] / rs.shift(126).dropna().iloc[-1] - 1.0) if len(rs)>=127 else None
            out["rs_12m"] = float(rs.iloc[-1] / rs.shift(252).dropna().iloc[-1] - 1.0) if len(rs)>=253 else None

    df = pd.DataFrame({"Date": close.index, "Close": close.values, "SMA50": sma50.values, "SMA200": sma200.values})
    return out, df

def simple_score(core: Core, mom: Dict[str,float]) -> Tuple[float, Dict[str,float], str]:
    parts={}
    v=0.0
    if core.pe and core.pe>0: v += max(0, min(30, (30 - core.pe)))/30*30
    if core.ev_ebitda and core.ev_ebitda>0: v += max(0, min(20, (20 - core.ev_ebitda)))/20*20
    if core.fcf_yield: v += max(0, min(25, core.fcf_yield*100))
    parts["Value"]=v
    q=0.0
    if core.roic: q += max(0, min(25, core.roic*100))
    if core.de is not None: q += max(0, min(10, (2.0 - core.de)*5)) if core.de<=2 else 0
    if core.int_cov: q += max(0, min(10, (core.int_cov/10)*10))
    parts["Quality"]=q
    m=0.0
    for key,weight in [("ret_12m",10),("ret_6m",10),("ret_3m",5)]:
        if mom.get(key) is not None: m += max(0, min(weight, mom[key]*100))
    if mom.get("sma50") and mom.get("sma200"): m += 10 if mom["sma50"]>=mom["sma200"] else 0
    parts["Momentum"]=m
    score = min(100.0, sum(parts.values()))
    verdict = "BUY" if score>=65 else ("HOLD" if score>=45 else "SELL")
    return score, parts, verdict

def piotroski_f(inc: pd.DataFrame, bal: pd.DataFrame, cfs: pd.DataFrame):
    details=[]; available=0; score=0
    ni_t, ni_p = _latest_prev(inc, ["net income"])
    assets_t, assets_p = _latest_prev(bal, ["total assets"])
    cfo_t, cfo_p = _latest_prev(cfs, ["operating cash flow","total cash from operating activities"])
    long_debt_t, long_debt_p = _latest_prev(bal, ["long term debt","long-term debt","total debt"])
    curr_as_t, curr_as_p = _latest_prev(bal, ["total current assets"])
    curr_li_t, curr_li_p = _latest_prev(bal, ["total current liabilities"])
    gross_t, gross_p = _latest_prev(inc, ["gross profit"])
    sales_t, sales_p = _latest_prev(inc, ["total revenue","revenue"])
    if ni_t is not None and assets_t: available+=1; ok=(ni_t/assets_t)>0; score+=int(ok); details.append(("ROA positive",ok))
    if cfo_t is not None: available+=1; ok=cfo_t>0; score+=int(ok); details.append(("CFO positive",ok))
    if cfo_t is not None and ni_t is not None: available+=1; ok=cfo_t>ni_t; score+=int(ok); details.append(("Accruals (CFO>NI)",ok))
    if long_debt_t is not None and long_debt_p is not None and assets_t:
        available+=1; ok=(long_debt_t/assets_t)<= (long_debt_p/(assets_p or assets_t)); score+=int(ok); details.append(("Leverage improving",ok))
    if curr_as_t and curr_li_t and curr_as_p and curr_li_p:
        available+=1; ok=(curr_as_t/curr_li_t) >= (curr_as_p/curr_li_p); score+=int(ok); details.append(("Current ratio improving",ok))
    if gross_t and sales_t and gross_p and sales_p:
        available+=1; ok=(gross_t/sales_t) >= (gross_p/sales_p); score+=int(ok); details.append(("Gross margin improving",ok))
    if sales_t and sales_p and assets_t and assets_p:
        available+=1; ok=(sales_t/assets_t) >= (sales_p/assets_p); score+=int(ok); details.append(("Asset turnover improving",ok))
    return score, available, details

# ========= Forensic models: Altman Z (variants) & Beneish M =========

def _is_financial_or_reit(sector: Optional[str], industry: Optional[str]) -> bool:
    s = (sector or "").lower(); i=(industry or "").lower()
    reit_kw = ["reit","real estate"]
    fin_kw  = ["financial", "bank", "insurance", "capital markets", "asset management", "mortgage"]
    return any(k in s for k in fin_kw) or any(k in i for k in fin_kw+reit_kw)

def _is_manufacturer(sector: Optional[str], industry: Optional[str]) -> bool:
    txt = f"{sector or ''} {industry or ''}".lower()
    manu_kw = [
        "manufactur","machinery","auto","semiconductor","hardware","electronics",
        "chemicals","metals","mining","materials","paper","textile","aerospace",
        "pharmaceutical","biotechnology","equipment","industrial","construction materials"
    ]
    return any(k in txt for k in manu_kw)

def choose_altman_variant(sector, industry):
    if _is_financial_or_reit(sector, industry):
        return "NA", "Not applicable for Financials/REITs"
    if _is_manufacturer(sector, industry):
        return "Z", "Public manufacturer (uses market value of equity)"
    # If you specifically want private manufacturer: change here to "Z'"
    return "Z''", "Non-manufacturer (emerging markets variant)"

def altman_z(core: Core, inc: pd.DataFrame, bal: pd.DataFrame) -> Dict:
    variant, note = choose_altman_variant(core.sector, core.industry)
    if variant == "NA":
        return {"variant": variant, "note": note, "score": None, "class": "n/a", "components": {}}

    ta = _latest_value(bal, ["total assets"])
    if not ta or ta<=0:
        return {"variant": variant, "note": note, "score": None, "class": "insufficient data", "components": {}}

    wc = None
    ca = _latest_value(bal, ["total current assets"])
    cl = _latest_value(bal, ["total current liabilities"])
    if ca is not None and cl is not None: wc = ca - cl
    re = _latest_value(bal, ["retained earnings"])
    ebit = _latest_value(inc, ["ebit","operating income"])
    sales = _latest_value(inc, ["total revenue","revenue"])

    total_liab = _latest_value(bal, ["total liab","total liabilities","total liability"])
    equity = _latest_value(bal, ["total stockholder","total shareholders'","total equity"])

    mve = None
    if core.mktcap: mve = core.mktcap
    elif core.price and core.shares_out: mve = core.price * core.shares_out

    comps = {
        "X1": (wc/ta) if (wc is not None) else None,
        "X2": (re/ta) if (re is not None) else None,
        "X3": (ebit/ta) if (ebit is not None) else None,
        "X4_mkt": (mve/total_liab) if (mve and total_liab) else None,
        "X4_book": (equity/total_liab) if (equity and total_liab) else None,
        "X5": (sales/ta) if (sales is not None) else None,
    }

    score = None; klass="unknown"
    if variant == "Z":
        need = [comps["X1"], comps["X2"], comps["X3"], comps["X4_mkt"], comps["X5"]]
        if all(v is not None for v in need):
            score = 1.2*comps["X1"] + 1.4*comps["X2"] + 3.3*comps["X3"] + 0.6*comps["X4_mkt"] + 1.0*comps["X5"]
            klass = "safe" if score>2.99 else ("grey" if score>=1.81 else "distress")
    elif variant == "Z''":
        need = [comps["X1"], comps["X2"], comps["X3"], comps["X4_book"]]
        if all(v is not None for v in need):
            score = 6.56*comps["X1"] + 3.26*comps["X2"] + 6.72*comps["X3"] + 1.05*comps["X4_book"]
            klass = "safe" if score>2.60 else ("grey" if score>=1.10 else "distress")
    else:  # Z' (private manufacturer) — not auto-selected but available
        need = [comps["X1"], comps["X2"], comps["X3"], comps["X4_book"], comps["X5"]]
        if all(v is not None for v in need):
            score = 0.717*comps["X1"] + 0.847*comps["X2"] + 3.107*comps["X3"] + 0.420*comps["X4_book"] + 0.998*comps["X5"]
            klass = "safe" if score>2.90 else ("grey" if score>=1.23 else "distress")

    return {"variant": variant, "note": note, "score": score, "class": klass, "components": comps}

def _dep_series(inc: pd.DataFrame, cfs: pd.DataFrame) -> Optional[pd.Series]:
    s = _series(inc, ["depreciation","depreciation & amortization","depreciation and amortization"])
    if s is None: s = _series(cfs, ["depreciation","depreciation & amortization"])
    return s

def beneish_mscore(inc: pd.DataFrame, bal: pd.DataFrame, cfs: pd.DataFrame) -> Dict:
    def vpair(df, names):
        t, p = _latest_prev(df, names)
        return t, p

    ar_t, ar_p = vpair(bal, ["accounts receivable"])
    sales_t, sales_p = vpair(inc, ["total revenue","revenue"])
    gp_t, gp_p = vpair(inc, ["gross profit"])
    ta_t, ta_p = vpair(bal, ["total assets"])
    ca_t, ca_p = vpair(bal, ["total current assets"])
    ppe_s = _series(bal, ["property plant equipment net","property, plant and equipment net","net property plant equipment","net ppe"])

    dep_s = _dep_series(inc, cfs)
    sga_t, sga_p = vpair(inc, ["selling general administrative","selling general and administrative","sg&a"])
    debt_t, debt_p = vpair(bal, ["total debt","long term debt","long-term debt"])
    cfo_t, cfo_p = vpair(cfs, ["operating cash flow","total cash from operating activities"])
    ni_t, ni_p = vpair(inc, ["net income"])

    def sdiv(a,b):
        try:
            if a is None or b in (None,0): return None
            return a/b
        except Exception: return None

    IDX = {}
    IDX["DSRI"] = sdiv(sdiv(ar_t, sales_t), sdiv(ar_p, sales_p))
    gm_t = sdiv(gp_t, sales_t); gm_p = sdiv(gp_p, sales_p)
    IDX["GMI"] = sdiv(gm_p, gm_t)
    def aqi(num_ta, num_ca, ppe):
        if num_ta is None or num_ca is None: return None
        ppe_t = _safe_float(ppe.iloc[0]) if ppe is not None and len(ppe)>0 else None
        ppe_p = _safe_float(ppe.iloc[1]) if ppe is not None and len(ppe)>1 else None
        aqi_t = sdiv((num_ta - (num_ca or 0) - (ppe_t or 0)), num_ta)
        aqi_p = sdiv((ta_p - (ca_p or 0) - (ppe_p or 0)), ta_p) if (ta_p and ca_p) else None
        return sdiv(aqi_t, aqi_p) if (aqi_t is not None and aqi_p is not None) else None
    IDX["AQI"] = aqi(ta_t, ca_t, ppe_s)
    IDX["SGI"] = sdiv(sales_t, sales_p)
    def depi(dep_s, ppe):
        if dep_s is None or ppe is None or len(dep_s)<2 or len(ppe)<2: return None
        dep_t = _safe_float(dep_s.iloc[0]); dep_p = _safe_float(dep_s.iloc[1])
        ppe_t = _safe_float(ppe.iloc[0]); ppe_p = _safe_float(ppe.iloc[1])
        r_t = sdiv(dep_t, (dep_t + (ppe_t or 0))); r_p = sdiv(dep_p, (dep_p + (ppe_p or 0)))
        return sdiv(r_p, r_t) if (r_t is not None and r_p is not None) else None
    IDX["DEPI"] = depi(dep_s, ppe_s)
    IDX["SGAI"] = sdiv(sdiv(sga_t, sales_t), sdiv(sga_p, sales_p))
    IDX["LVGI"] = sdiv(sdiv(debt_t, ta_t), sdiv(debt_p, ta_p))
    IDX["TATA"] = sdiv((None if ni_t is None or cfo_t is None else (ni_t - cfo_t)), ta_t)

    w = {"DSRI":0.920,"GMI":0.528,"AQI":0.404,"SGI":0.892,"DEPI":0.115,"SGAI":-0.172,"TATA":4.679,"LVGI":-0.327}
    used = {k:v for k,v in IDX.items() if v is not None}
    n_avail = len(used)
    m = None
    if n_avail >= 6:
        m = -4.84 + sum(w[k]*used.get(k,0) for k in w.keys())
    klass = None
    if m is not None:
        klass = "flagged" if m > -1.78 else "not flagged"
    return {"m_score": m, "class": klass, "indices": IDX, "n_avail": n_avail}

# ========= Smart flags, issues, DCF, external signals, recommendation =========

def smart_flags(core: Core, inc: pd.DataFrame, bal: pd.DataFrame, cfs: pd.DataFrame, mom: Dict[str,float]):
    flags=[]
    def add(level,title,detail): flags.append({"level":level,"title":title,"detail":detail})
    if core.revenue is None or core.ebit is None or core.equity is None:
        add("info","Missing key lines","Some statements lacked rows (Revenue/EBIT/Equity). Trend scores may be limited.")
    cfo = _latest_value(cfs, ["operating cash flow","total cash from operating activities"])
    if core.net_income and cfo is not None and cfo < 0 <= core.net_income:
        add("red","Earnings quality risk","Positive net income but negative operating cash flow in latest period.")
    ar_t, ar_p = _latest_prev(bal, ["accounts receivable"])
    sales_t, sales_p = _latest_prev(inc, ["total revenue","revenue"])
    if ar_t and sales_t and ar_p and sales_p and sales_p!=0 and sales_t!=0:
        dsri = (ar_t/sales_t) / (ar_p/sales_p)
        if dsri > 1.4: add("orange","Receivables outpacing sales","DSRI > 1.4; watch for aggressive revenue recognition.")
    inv_t, inv_p = _latest_prev(bal, ["inventory"])
    if inv_t and inv_p and sales_t and sales_p and sales_p!=0:
        inv_growth = (inv_t-inv_p)/abs(inv_p) if inv_p!=0 else None
        sales_growth = (sales_t-sales_p)/abs(sales_p) if sales_p!=0 else None
        if inv_growth is not None and sales_growth is not None and inv_growth > sales_growth*1.5 and inv_growth>0.3:
            add("orange","Inventory build-up","Inventory growing much faster than sales; potential obsolescence risk.")
    assets = _latest_value(bal, ["total assets"])
    if core.goodwill and assets and core.goodwill/assets > 0.4:
        add("orange","High goodwill share of assets",">40% of assets are goodwill; future impairments possible.")
    if core.current_ratio and core.current_ratio < 1.0: add("orange","Tight liquidity","Current ratio < 1.0")
    if core.quick_ratio and core.quick_ratio < 1.0: add("orange","Low quick ratio","Quick ratio < 1.0")
    if core.de and core.de > 2.0: add("orange","High leverage","Debt/Equity > 2")
    if core.int_cov and core.int_cov < 2.0: add("orange","Weak interest coverage","EBIT/Interest < 2")
    gm = (core.gross_profit/core.revenue) if (core.gross_profit and core.revenue) else None
    if gm is not None and gm < 0.15: add("orange","Thin gross margin","Gross margin < 15%")
    if core.ebitda and core.revenue and core.ebitda/core.revenue < 0.05:
        add("orange","Low EBITDA margin","EBITDA margin < 5%")
    if mom.get("max_dd_1y") is not None and mom["max_dd_1y"] < -0.3:
        add("orange","Deep drawdown","Max drawdown worse than -30% in last year.")
    if mom.get("vol_1y") is not None and mom["vol_1y"] > 0.6:
        add("info","Very high volatility","Annualized volatility > 60%")
    return flags

def collect_data_issues(info: dict, inc: pd.DataFrame, bal: pd.DataFrame, cfs: pd.DataFrame) -> List[str]:
    issues=[]
    def chk(df, names, label):
        if _series(df, names) is None: issues.append(f"Missing: {label}")
    chk(inc, ["total revenue","revenue"], "Revenue (IS)")
    chk(inc, ["gross profit"], "Gross Profit (IS)")
    chk(inc, ["ebit","operating income"], "EBIT (IS)")
    chk(inc, ["net income"], "Net Income (IS)")
    chk(bal, ["total stockholder","total shareholders'","total equity"], "Shareholders' Equity (BS)")
    chk(bal, ["total assets"], "Total Assets (BS)")
    chk(bal, ["total debt","long term debt","long-term debt"], "Total Debt (BS)")
    chk(cfs, ["operating cash flow","total cash from operating activities"], "Operating Cash Flow (CF)")
    chk(cfs, ["free cash flow","free cash flow to the firm"], "Free Cash Flow (CF)")
    return issues

def simple_dcf(core: Core, rf: float, mkt_prem: float, term_growth: float, years: int=5):
    beta = core.beta or 1.0; re = rf + beta*mkt_prem; rd = rf + 0.015
    E = core.mktcap or 0; D = core.debt or 0
    if E+D==0: wacc = re
    else: wacc = (E/(E+D))*re + (D/(E+D))*rd*(1-0.21)
    fcf0 = core.fcf if core.fcf else (core.net_income or 0)
    if not fcf0 or fcf0<=0: fcf0 = (core.ebit or 0)*0.79
    g1 = min(0.12, max(0.00, mkt_prem)); gs = np.linspace(g1, term_growth, years)
    fcf_list=[]; pv=0.0
    for t,g in enumerate(gs, start=1):
        fcf_t = fcf0*((1+g)**t); fcf_list.append(fcf_t); pv += fcf_t/((1+wacc)**t)
    tv = fcf_list[-1]*(1+term_growth)/(wacc-term_growth) if wacc>term_growth else 0.0
    pv += tv/((1+wacc)**years)
    net_debt = (core.debt or 0) - (core.cash or 0); eq_val = pv - net_debt
    per_share = eq_val / core.shares_out if core.shares_out else None
    df = pd.DataFrame({"Year": list(range(1,years+1)),
                       "Growth": [f"{g:.2%}" for g in gs],
                       "FCF": fcf_list,
                       "DF": [1/((1+wacc)**t) for t in range(1,years+1)]})
    return df, per_share or 0.0

# ==============
# External signals
# ==============

@st.cache_data(ttl=1800, show_spinner=False)
def news_sentiment(ticker: str, days: int=14) -> Optional[float]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        return None
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", None) or []
        if not news: return None
        cutoff = pd.Timestamp.utcnow().timestamp() - days*24*3600
        titles = [n.get("title") for n in news if n.get("providerPublishTime",0) >= cutoff]
        if not titles: return None
        vs = SentimentIntensityAnalyzer()
        scores = [vs.polarity_scores(tt)["compound"] for tt in titles if tt]
        return float(np.mean(scores)) if scores else None
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def analyst_signals(ticker: str, info_fallback: dict) -> Dict:
    out = {}
    t = yf.Ticker(ticker)
    rec_df = None
    try:
        if hasattr(t, "get_recommendations"):
            rec_df = t.get_recommendations()
        elif hasattr(t, "recommendations"):
            rec_df = t.recommendations
    except Exception:
        rec_df = None
    if isinstance(rec_df, pd.DataFrame) and not rec_df.empty:
        try:
            recent = rec_df.sort_index().last("90D")
            if not recent.empty and "To Grade" in recent:
                tg = recent["To Grade"].astype(str)
                out["recs_90d"] = {
                    "buy": int(tg.str.contains("Buy", case=False).sum()),
                    "hold": int(tg.str.contains("Hold", case=False).sum()),
                    "sell": int(tg.str.contains("Sell", case=False).sum()),
                }
        except Exception:
            pass
    for k in ["targetMeanPrice","targetLowPrice","targetHighPrice","numberOfAnalystOpinions","recommendationMean","recommendationKey"]:
        if info_fallback.get(k) is not None:
            out[k] = info_fallback.get(k)
    return out

@st.cache_data(ttl=900, show_spinner=False)
def options_signals(ticker: str, last_price: Optional[float]) -> Optional[Dict[str,float]]:
    if not last_price: return None
    try:
        t = yf.Ticker(ticker)
        exps = t.options or []
        if not exps: return None
        oc = t.option_chain(exps[0])
        calls, puts = getattr(oc, "calls", None), getattr(oc, "puts", None)
        if calls is None or puts is None or calls.empty or puts.empty:
            return None
        idx_c = (calls["strike"]-last_price).abs().idxmin()
        idx_p = (puts["strike"]-last_price).abs().idxmin()
        atm_iv = float(np.nanmean([calls.loc[idx_c,"impliedVolatility"], puts.loc[idx_p,"impliedVolatility"]]))
        pcr_oi = float(puts["openInterest"].sum() / max(1, calls["openInterest"].sum()))
        pcr_vol = float(puts["volume"].sum() / max(1, calls["volume"].sum()))
        k_dn = last_price*0.9; k_up = last_price*1.1
        iv_dn = float(puts.iloc[(puts["strike"]-k_dn).abs().idxmin()]["impliedVolatility"])
        iv_up = float(calls.iloc[(calls["strike"]-k_up).abs().idxmin()]["impliedVolatility"])
        skew10 = iv_dn - atm_iv
        return {"atm_iv": atm_iv, "pcr_oi": pcr_oi, "pcr_vol": pcr_vol, "skew10": skew10}
    except Exception:
        return None

def short_interest(info: dict) -> Dict[str, Optional[float]]:
    return {
        "sharesShort": _safe_float(info.get("sharesShort")),
        "shortPercentFloat": _safe_float(info.get("shortPercentOfFloat")),
        "daysToCover": _safe_float(info.get("shortRatio")),
    }

# ==============
# Recommendation engine (with signals & forensic overrides)
# ==============

def multiples_fair_values(core: Core, peer_meds: Dict[str,float]):
    out = {}
    eps = (core.net_income or 0) / (core.shares_out or 1) if core.shares_out else None
    if eps and peer_meds.get("P/E"):
        out["PE"] = eps * peer_meds["P/E"]
    if core.ebitda and peer_meds.get("EV/EBITDA"):
        ev = core.ebitda * peer_meds["EV/EBITDA"]
        eq = ev - ((core.debt or 0) - (core.cash or 0))
        if core.shares_out:
            out["EVEBITDA"] = eq / core.shares_out
    return out

def combine_fair_value(dcf_px: Optional[float], mults_dict: Dict[str,float], roic: Optional[float]=None):
    vals = [v for v in [dcf_px] + list(mults_dict.values()) if v]
    if not vals:
        return None, None, None
    lo, hi = min(vals), max(vals)
    mean = float(np.median(vals))
    return lo, mean, hi

def adjust_mos(base_mos: float,
               sentiment: Optional[float],
               analyst: Dict,
               opt_sig: Optional[Dict[str,float]],
               si: Dict[str,Optional[float]],
               price: Optional[float],
               beneish_flag: bool,
               altman_distress: bool) -> float:
    mos = base_mos
    if sentiment is not None:
        if sentiment >= 0.2: mos = max(0.05, mos - 0.03)
        elif sentiment <= -0.2: mos = min(0.4, mos + 0.05)
    if analyst:
        rec_mean = _safe_float(analyst.get("recommendationMean"))
        tgt = _safe_float(analyst.get("targetMeanPrice"))
        prem = (tgt / price - 1.0) if (price and tgt) else None
        if rec_mean is not None:
            if rec_mean <= 2.5 and (prem is None or prem >= 0.05):
                mos = max(0.05, mos - 0.02)
            if rec_mean >= 3.5 or (prem is not None and prem <= -0.10):
                mos = min(0.5, mos + 0.03)
    if opt_sig:
        if opt_sig.get("pcr_oi", 0) > 1.5: mos = min(0.5, mos + 0.03)
        if opt_sig.get("atm_iv", 0) > 0.6: mos = min(0.5, mos + 0.03)
        if opt_sig.get("skew10", 0) > 0.10: mos = min(0.5, mos + 0.02)
        if opt_sig.get("pcr_oi", 0) < 0.7 and opt_sig.get("skew10", 0) < -0.05:
            mos = max(0.05, mos - 0.02)
    if si:
        if (si.get("shortPercentFloat") or 0) > 0.15: mos = min(0.6, mos + 0.05)
        if (si.get("daysToCover") or 0) > 5: mos = min(0.6, mos + 0.03)
    # Forensic overrides
    if beneish_flag: mos = min(0.65, mos + 0.05)
    if altman_distress: mos = min(0.65, mos + 0.05)
    return mos

def recommend_bands(core: Core,
                    dcf_px: Optional[float],
                    peer_meds: Dict[str,float],
                    mom: Dict[str,float],
                    sentiment: Optional[float],
                    analyst: Dict,
                    opt_sig: Optional[Dict[str,float]],
                    si: Dict[str,Optional[float]],
                    base_mos=0.20, trim=0.15, red_flags=False,
                    beneish_flag=False, altman_distress=False):
    mults = multiples_fair_values(core, peer_meds or {})
    lo, fair, hi = combine_fair_value(dcf_px, mults, core.roic)
    if fair is None:
        return None
    mos = base_mos + (0.10 if red_flags else 0.0)
    mos = adjust_mos(mos, sentiment, analyst, opt_sig, si, core.price, beneish_flag, altman_distress)
    buy_px  = fair * (1 - mos)
    sell_px = fair * (1 + trim)

    timing_ok = False
    if mom:
        timing_ok = ((mom.get("sma50") and mom.get("sma200") and mom["sma50"] >= mom["sma200"]) or
                     (mom.get("price") and mom.get("sma50") and mom["price"] >= mom["sma50"]) or
                     (mom.get("rs_6m") is not None and mom["rs_6m"] > 0))
    return {
        "fair_low": lo, "fair_mid": fair, "fair_high": hi,
        "buy_below": buy_px, "trim_above": sell_px,
        "timing_ok": timing_ok,
        "used_mults": mults,
        "mos": mos
    }

# ==============
# Analyze wrapper
# ==============

def analyze_one(ticker: str, hist_period: str, bench: str, rf: float, mkt_prem: float, term_g: float):
    info, inc, bal, cfs, hist = fetch_all(ticker, hist_period)
    bench_hist = fetch_history_only(bench, hist_period)
    core = build_core(ticker, info, inc, bal, cfs)
    mom, price_df = momentum_and_prices(hist, bench_hist)
    score, parts, verdict = simple_score(core, mom)
    dcf_table, dcf_px = simple_dcf(core, rf, mkt_prem, term_g, years=5)
    f_score, f_avail, f_details = piotroski_f(inc, bal, cfs)
    flags = smart_flags(core, inc, bal, cfs, mom)
    issues = collect_data_issues(info, inc, bal, cfs)
    cap = {"Market Cap": core.mktcap, "Enterprise Value": (core.ev or None), "Debt": core.debt,
           "Cash": core.cash, "Net Debt": (core.debt or 0)-(core.cash or 0), "Shares Out": core.shares_out}
    # New forensic models
    alt = altman_z(core, inc, bal)
    ben = beneish_mscore(inc, bal, cfs)
    return core, dcf_table, mom, price_df, score, parts, verdict, cap, dcf_px, f_score, f_avail, f_details, flags, issues, info, alt, ben

# ============================
# UI (Run-gated & sticky summary)
# ============================

st.title("🧠 Naked Fundamentals PRO — Analyst Dashboard")
st.caption("Fundamentals • V/Q/M • DCF • Forensics (Piotroski, Altman Z, Beneish M) • Signals • Export")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Single Ticker","Compare 3"], index=0,
                    help="Single security deep dive or side-by-side comparison.")
    hist_period = st.selectbox("Price History Period", ["1y","3y","5y"], index=0,
                               help="Span for momentum & MAs.")
    bench = st.selectbox("Benchmark (for RS)", ["SPY","QQQ","IWM"], index=0,
                         help="Relative strength is computed vs this benchmark.")
    if mode=="Single Ticker":
        tk1 = st.text_input("Ticker", "AAPL", help="Symbol on Yahoo Finance.").upper().strip()
    else:
        tk1 = st.text_input("Ticker 1", "AAPL").upper().strip()
        tk2 = st.text_input("Ticker 2", "MSFT").upper().strip()
        tk3 = st.text_input("Ticker 3", "GOOGL").upper().strip()

    st.markdown("---")
    st.subheader("Valuation Settings")
    rf = st.number_input("Risk-free rate (rf)", value=0.045, step=0.005, format="%.3f",
                         help="Annual risk-free rate used in CAPM/WACC.")
    mkt_prem = st.number_input("Market risk premium", value=0.055, step=0.005, format="%.3f",
                               help="Equity risk premium × beta is added to rf.")
    term_g = st.number_input("Terminal growth", value=0.020, step=0.002, format="%.3f",
                             help="Long-run growth for terminal value.")

    st.markdown("---")
    peer_str = st.text_input("Peers (optional)", "",
                             help="Comma-separated list for peer medians (P/E, EV/EBITDA, ROIC, etc.).")

    st.markdown("---")
    st.subheader("External Signals")
    use_news = st.checkbox("News sentiment (VADER)", value=True, help="Scores recent Yahoo headlines (−1 to +1).")
    use_analyst = st.checkbox("Analyst consensus & targets", value=True, help="From yfinance info + recommendations feed.")
    use_options = st.checkbox("Options / IV signals", value=True, help="ATM IV, put/call ratios, 10% OTM skew.")
    use_short = st.checkbox("Short interest", value=True, help="Short % of float and days-to-cover (if available).")

    st.markdown("---")
    colb1, colb2 = st.columns(2)
    run_clicked = colb1.button("▶️ Run analysis", type="primary", help="Click to fetch & compute.")
    if colb2.button("♻️ Hard refresh (clear cache)", help="Clears Streamlit cache."):
        st.cache_data.clear()
        st.success("Cache cleared. Click Run analysis again.")

    try:
        import sys
        st.caption(f"Py {sys.version.split()[0]} • pandas {pd.__version__} • numpy {np.__version__} • yfinance {getattr(yf,'__version__','?')}")
    except Exception:
        pass

# Sticky summary CSS
st.markdown("""
<style>
.sticky-summary { position: sticky; top: 0; z-index: 999; padding: 0.5rem 0.75rem;
                  background: rgba(250, 250, 250, 0.95); backdrop-filter: blur(2px);
                  border-bottom: 1px solid #eee; }
.sticky-kpi { display: inline-block; margin-right: 1.5rem; font-weight: 600; }
.sticky-kpi span { font-weight: 400; color: #333; }
</style>
""", unsafe_allow_html=True)

def _summary_bar(ticker, name, price, score, verdict):
    st.markdown(f"""
<div class="sticky-summary">
  <div class="sticky-kpi" title="Yahoo last trade / fast_info">{ticker} — {name}</div>
  <div class="sticky-kpi" title="Last price from data source">Price: <span>{_num_suffix(price)}</span></div>
  <div class="sticky-kpi" title="Composite of Value/Quality/Momentum (0–100)">Score: <span>{score:.1f}</span></div>
  <div class="sticky-kpi" title="BUY≥65 • HOLD≥45 • else SELL">Verdict: <span>{verdict}</span></div>
</div>
""", unsafe_allow_html=True)

# ==============
# Main panes
# ==============

if mode=="Single Ticker":
    if not run_clicked:
        st.info("Set your inputs, then click **Run analysis**.")
        st.stop()

    if not tk1:
        st.warning("Enter a ticker.")
        st.stop()

    started = pd.Timestamp.utcnow()
    with st.spinner(f"Fetching {tk1}..."):
        (core, dcf_table, mom, price_df, score, parts, verdict, cap, dcf_px,
         f_score, f_avail, f_details, flags, issues, info, alt, ben) = analyze_one(
            tk1, hist_period, bench, rf, mkt_prem, term_g
        )

    _summary_bar(tk1, core.name or "", core.price, score, verdict)
    st.caption(f"Source: yfinance • Retrieved at {started.strftime('%Y-%m-%d %H:%M')} UTC")

    # Headline metrics
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Price", _num_suffix(core.price))
    colB.metric("Market Cap", _num_suffix(core.mktcap))
    colC.metric("Beta", _num_suffix(core.beta))
    colD.metric("Dividend Yield", _num_suffix(core.dividend_yield, pct=True))

    # Score breakdown
    st.markdown("### Score & Verdict")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{score:.1f}/100")
    c2.metric("Value", f"{parts.get('Value',0):.1f}")
    c3.metric("Quality", f"{parts.get('Quality',0):.1f}")
    c4.metric("Momentum", f"{parts.get('Momentum',0):.1f}")
    st.success("Verdict thresholds: BUY ≥ 65, HOLD ≥ 45, else SELL.")

    # Data issues panel
    with st.expander("Data issues & coverage", expanded=bool(issues)):
        if issues:
            st.warning("Some inputs were missing. Calculations that depend on them may show “—”.")
            st.write("\n".join(f"• {x}" for x in issues))
        else:
            st.info("No data gaps detected in key lines for the latest two periods.")

    # Price chart
    if not price_df.empty:
        st.markdown("### Price & Moving Averages")
        fig = px.line(price_df, x="Date", y=["Close","SMA50","SMA200"])
        fig.update_traces(mode="lines", hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}")
        st.plotly_chart(fig, use_container_width=True)

    # Momentum extras
    st.markdown("### Momentum Extras")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vol (ann.)", _num_suffix(mom.get("vol_1y")))
    m2.metric("Max DD (1y)", _num_suffix(mom.get("max_dd_1y"), pct=True))
    m3.metric(f"RS vs {bench} (6m)", _num_suffix(mom.get("rs_6m"), pct=True) if mom.get("rs_6m") is not None else "—")
    m4.metric(f"RS vs {bench} (12m)", _num_suffix(mom.get("rs_12m"), pct=True) if mom.get("rs_12m") is not None else "—")

    # Key Fundamentals with tooltips
    st.markdown("### Key Fundamentals")
    rows = [
        ("Sector", core.sector, "Company sector classification"),
        ("Industry", core.industry, "More granular industry group"),
        ("P/E", core.pe, "Price / Earnings (TTM approximation)"),
        ("EV/EBITDA", core.ev_ebitda, "Enterprise Value / EBITDA"),
        ("P/S", core.ps, "Market Cap / Revenue"),
        ("P/B", core.pb, "Market Cap / Book Equity"),
        ("Debt/Equity", core.de, "Leverage ratio"),
        ("Interest Coverage", core.int_cov, "EBIT / |Interest Expense|"),
        ("Current Ratio", core.current_ratio, "Current Assets / Current Liabilities"),
        ("Quick Ratio", core.quick_ratio, "(Current Assets – Inventory) / Current Liabilities"),
        ("FCF", core.fcf, "Free Cash Flow"),
        ("FCF Yield", core.fcf_yield, "FCF / Market Cap"),
        ("ROIC", core.roic, "NOPAT / Invested Capital (avg)"),
        ("Cash Conversion Cycle (days)", core.ccc, "AR days + Inventory days − AP days"),
    ]
    df_fund = pd.DataFrame({
        "Metric": [r[0] for r in rows],
        "Value": [_num_suffix(r[1], pct=("Yield" in r[0] or r[0]=="ROIC")) for r in rows],
        "ⓘ": [r[2] for r in rows],
    })
    st.dataframe(
        df_fund,
        use_container_width=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", help="Name of the ratio/line item."),
            "Value": st.column_config.TextColumn("Value", help="Formatted with K/M/B/T and % where relevant."),
            "ⓘ": st.column_config.TextColumn("What this means", help="Hover the cell for a quick explanation."),
        }
    )

    # Capital Allocation
    st.markdown("### Capital Allocation")
    cap_df = pd.DataFrame({"Metric": list(cap.keys()), "Value": [_num_suffix(v) for v in cap.values()]})
    st.dataframe(cap_df, use_container_width=True)

    # Forensics — Piotroski, Altman, Beneish
    st.markdown("### Forensics")
    ff1, ff2, ff3 = st.columns([1,1,2])

    # Piotroski
    ff1.metric("Piotroski F-Score", f"{f_score}/{f_avail}" if f_avail else "—")
    if f_avail:
        f_tbl = pd.DataFrame([{"Check": n, "Pass": "✅" if ok else "❌"} for n, ok in f_details])
        ff1.dataframe(f_tbl, use_container_width=True, hide_index=True)
    else:
        ff1.info("F-Score not available (insufficient history).")

    # Altman Z (variant shown)
    alt_score = alt.get("score")
    alt_variant = alt.get("variant")
    alt_class = alt.get("class")
    ff2.metric(f"Altman {alt_variant}", f"{alt_score:.2f}" if alt_score is not None else "—",
               help="Z (public manuf.): safe>2.99, grey 1.81–2.99, distress<1.81 • "
                    "Z′ (private manuf.): safe>2.90, grey 1.23–2.90 • "
                    "Z″ (non-manuf.): safe>2.60, grey 1.10–2.60")
    ff2.caption(alt.get("note",""))
    if alt_class == "distress":
        ff2.error("Distress zone")
    elif alt_class == "grey":
        ff2.warning("Grey zone")
    elif alt_class == "safe":
        ff2.success("Safe zone")

    # Beneish
    m = ben.get("m_score"); mclass = ben.get("class"); n_av = ben.get("n_avail",0)
    ff3.metric("Beneish M-Score", f"{m:.2f}" if m is not None else "—",
               help=">-1.78: higher manipulation risk. Uses 8 indices; needs ≥6 to compute.")
    ff3.caption(f"Indices available: {n_av}/8")
    if mclass == "flagged":
        ff3.error("Higher manipulation risk (M > -1.78)")
    elif mclass == "not flagged":
        ff3.success("Not flagged (≤ -1.78)")
    # Show indices table
    idx_tbl = pd.DataFrame([{"Index": k, "Value": v} for k,v in (ben.get("indices") or {}).items()])
    if not idx_tbl.empty:
        ff3.dataframe(idx_tbl, use_container_width=True, hide_index=True)

    # Smart Flags panel (existing)
    if flags:
        sev_map = {"red":"🔴","orange":"🟠","info":"🔵"}
        flags_df = pd.DataFrame([{"Severity": sev_map.get(f["level"],"🔵"), "Flag": f["title"], "Details": f["detail"]} for f in flags])
        st.dataframe(flags_df, use_container_width=True, hide_index=True)
    else:
        st.info("No red/orange smart flags detected with current data.")

    # Peer Medians (optional)
    peer_meds = {}
    if peer_str.strip():
        st.markdown("### Peer Medians")
        peers = sorted({p.strip().upper() for p in peer_str.split(",") if p.strip()})
        @st.cache_data(ttl=3600, show_spinner=False)
        def _peer_meds(peers_list: List[str]) -> Dict[str,float]:
            rows=[]
            for p in peers_list:
                info2, inc2, bal2, cfs2, _ = fetch_all(p, "1y")
                c2 = build_core(p, info2, inc2, bal2, cfs2)
                rows.append({"P/E": c2.pe, "EV/EBITDA": c2.ev_ebitda, "P/S": c2.ps, "P/B": c2.pb,
                             "Debt/Equity": c2.de, "FCF Yield": c2.fcf_yield, "ROIC": c2.roic})
            if not rows: return {}
            return pd.DataFrame(rows).median(numeric_only=True).to_dict()
        with st.spinner("Fetching peers..."):
            peer_meds = _peer_meds(peers)
        if peer_meds:
            bench_tbl = pd.DataFrame([
                {"Metric":"P/E", "Company":core.pe, "Peers (Median)":peer_meds.get("P/E")},
                {"Metric":"EV/EBITDA", "Company":core.ev_ebitda, "Peers (Median)":peer_meds.get("EV/EBITDA")},
                {"Metric":"P/S", "Company":core.ps, "Peers (Median)":peer_meds.get("P/S")},
                {"Metric":"P/B", "Company":core.pb, "Peers (Median)":peer_meds.get("P/B")},
                {"Metric":"Debt/Equity", "Company":core.de, "Peers (Median)":peer_meds.get("Debt/Equity")},
                {"Metric":"FCF Yield", "Company":core.fcf_yield, "Peers (Median)":peer_meds.get("FCF Yield")},
                {"Metric":"ROIC", "Company":core.roic, "Peers (Median)":peer_meds.get("ROIC")},
            ])
            st.dataframe(bench_tbl, use_container_width=True)
        else:
            st.info("No peer data available right now (source may be rate-limiting).")

    # DCF
    st.markdown("### DCF (simple)")
    st.dataframe(dcf_table, use_container_width=True)
    if dcf_px:
        st.info(f"**DCF Fair Value (approx)**: {_num_suffix(dcf_px)} per share")

    # ===== External Signals block =====
    st.markdown("### External Signals")
    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)

    sentiment = news_sentiment(tk1) if use_news else None
    if use_news:
        sig_col1.metric("News Sentiment (14d)", f"{sentiment:.2f}" if sentiment is not None else "—",
                        help="VADER compound score averaged over recent Yahoo headlines (−1..+1).")

    analyst = analyst_signals(tk1, info) if use_analyst else {}
    if use_analyst:
        rec_mean = analyst.get("recommendationMean")
        tgt = analyst.get("targetMeanPrice")
        sig_col2.metric("Analyst Consensus", f"{rec_mean:.2f}" if rec_mean is not None else "—",
                        help="1=Strong Buy … 5=Sell")
        sig_col2.metric("Target (mean)", _num_suffix(tgt) if tgt else "—")

    opt_sig = options_signals(tk1, core.price) if use_options else None
    if use_options:
        sig_col3.metric("ATM IV (near exp.)", _num_suffix(opt_sig.get("atm_iv"), pct=True) if opt_sig else "—",
                        help="Implied vol at strike nearest spot.")
        sig_col3.metric("Put/Call OI", f"{opt_sig.get('pcr_oi'):.2f}" if opt_sig else "—",
                        help="Put open interest ÷ Call open interest (near expiry).")

    si = short_interest(info) if use_short else {}
    if use_short:
        spf = si.get("shortPercentFloat")
        dtc = si.get("daysToCover")
        sig_col4.metric("Short % Float", _num_suffix(spf, pct=True) if spf is not None else "—")
        sig_col4.metric("Days-to-Cover", f"{dtc:.2f}" if dtc is not None else "—",
                        help="Short interest / average daily volume.")

    # ===== Recommendation bands =====
    st.markdown("### Recommendation")
    red_present = any(fl["level"]=="red" for fl in (flags or []))
    beneish_flag = (ben.get("class") == "flagged")
    altman_distress = (alt.get("class") == "distress")
    bands = recommend_bands(core, dcf_px, peer_meds, mom, sentiment if use_news else None,
                            analyst if use_analyst else {}, opt_sig if use_options else None,
                            si if use_short else {}, base_mos=0.20, trim=0.15,
                            red_flags=red_present, beneish_flag=beneish_flag, altman_distress=altman_distress)
    if bands:
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Fair Low", _num_suffix(bands["fair_low"]))
        b2.metric("Fair Mid", _num_suffix(bands["fair_mid"]))
        b3.metric("Fair High", _num_suffix(bands["fair_high"]))
        b4.metric("Buy Below", _num_suffix(bands["buy_below"]), help=f"Margin of Safety ≈ {bands['mos']:.0%}")
        b5.metric("Trim Above", _num_suffix(bands["trim_above"]))
        msg = []
        if core.price and bands["buy_below"] and core.price < bands["buy_below"]:
            msg.append("Price in **BUY** zone.")
        if core.price and bands["trim_above"] and core.price > bands["trim_above"]:
            msg.append("Price in **TRIM/SELL** zone.")
        if not bands["timing_ok"]:
            msg.append("Timing filter **not met** (SMA/RS).")
        if red_present or beneish_flag or altman_distress:
            msg.append("Forensic/quality risks → stricter MoS applied.")
        st.info(" • ".join(msg) if msg else "Neutral: between Buy and Trim bands.")
    else:
        st.info("Insufficient data to form fair-value bands (need DCF or multiples).")

    # Sensitivity grid
    rows=[]
    for prem in [0.045,0.055,0.065]:
        for g in [0.015,0.020,0.025]:
            _, ps = simple_dcf(core, rf, prem, g, years=5)
            rows.append({"MktPrem": prem, "TerminalG": g, "Price/Share": ps})
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
            if f_avail:
                ftbl = pd.DataFrame([{"Check": n, "Pass": bool(ok)} for n, ok in f_details])
                sanitize_for_excel(ftbl).to_excel(writer, index=False, sheet_name="Piotroski")
            if flags:
                sev_map = {"red":"RED","orange":"ORANGE","info":"INFO"}
                flag_df = pd.DataFrame([{"Severity": sev_map.get(f["level"],"INFO"), "Flag": f["title"], "Details": f["detail"]} for f in flags])
                sanitize_for_excel(flag_df).to_excel(writer, index=False, sheet_name="Flags")
            if not price_df.empty:
                pdf = price_df.copy()
                if "Date" in pdf.columns and pd.api.types.is_datetime64_any_dtype(pdf["Date"]):
                    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce").dt.tz_localize(None)
                sanitize_for_excel(pdf).to_excel(writer, index=False, sheet_name="Prices")
            # Signals & Recommendation sheet
            sig_rows = []
            if use_news: sig_rows.append({"Signal":"News Sentiment (14d)","Value": sentiment})
            if use_analyst:
                for k,v in analyst.items():
                    sig_rows.append({"Signal": f"Analyst {k}", "Value": v})
            if use_options and opt_sig:
                for k,v in opt_sig.items():
                    sig_rows.append({"Signal": f"Options {k}", "Value": v})
            if use_short and si:
                for k,v in si.items():
                    sig_rows.append({"Signal": f"Short {k}", "Value": v})
            if sig_rows:
                sanitize_for_excel(pd.DataFrame(sig_rows)).to_excel(writer, index=False, sheet_name="Signals")
            # Altman & Beneish sheet
            alt_rows = [{"Metric":"Altman Variant", "Value": alt_variant},
                        {"Metric":"Altman Score", "Value": alt_score},
                        {"Metric":"Altman Class", "Value": alt_class},
                        {"Metric":"Beneish M-Score", "Value": m},
                        {"Metric":"Beneish Class", "Value": mclass},
                        {"Metric":"Beneish Indices Available", "Value": ben.get("n_avail",0)}]
            sanitize_for_excel(pd.DataFrame(alt_rows)).to_excel(writer, index=False, sheet_name="Altman_Beneish")
        st.download_button(
            "Download Excel",
            bio.getvalue(),
            file_name=f"{tk1}_Naked_PRO.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    # Compare 3 (run-gated) — unchanged summary view
    if not run_clicked:
        st.info("Enter tickers, then click **Run analysis**.")
        st.stop()
    tickers = [t for t in [tk1, tk2, tk3] if t]
    if not tickers:
        st.warning("Enter at least one ticker.")
        st.stop()

    st.subheader("Compare 3 — Summary")
    results=[]; charts=[]
    def _job(tk):
        try:
            return tk, analyze_one(tk, hist_period, bench, rf, mkt_prem, term_g)
        except Exception as e:
            return tk, e

    with st.spinner("Fetching tickers in parallel..."):
        with ThreadPoolExecutor(max_workers=min(3, len(tickers))) as ex:
            futs=[ex.submit(_job, tk) for tk in tickers]
            for f in as_completed(futs):
                tk, res = f.result()
                if isinstance(res, Exception):
                    st.error(f"{tk}: {res}"); continue
                core, dcf_table, mom, price_df, score, parts, verdict, cap, dcf_px, f_score, f_avail, f_details, flags, issues, info, alt, ben = res
                results.append({
                    "Ticker": tk, "Name": core.name, "Price": _num_suffix(core.price),
                    "Score": round(score,1), "Verdict": verdict, "P/E": _num_suffix(core.pe),
                    "EV/EBITDA": _num_suffix(core.ev_ebitda), "D/E": _num_suffix(core.de),
                    "FCF Yield": _num_suffix(core.fcf_yield, pct=True), "ROIC": _num_suffix(core.roic, pct=True),
                    "F-Score": f"{f_score}/{f_avail}" if f_avail else "—",
                    "Flags (red/orange)": sum(1 for fl in (flags or []) if fl["level"] in ("red","orange")),
                    "Altman": f"{alt.get('variant')}:{alt.get('class') if alt.get('class') else '—'}",
                    "Beneish": ben.get("class") or "—"
                })
                if not price_df.empty: charts.append((tk, price_df))

    if results:
        cmp_df = pd.DataFrame(results)
        st.dataframe(cmp_df, use_container_width=True)

        if charts:
            st.markdown("### Normalized Prices (last = 100)")
            parts=[]
            for tk, dfp in charts:
                ser = dfp["Close"].dropna()
                if ser.empty: continue
                base = ser.iloc[-1]
                parts.append(pd.DataFrame({"Date": dfp["Date"], "Ticker": tk, "Index": ser/base*100}))
            if parts:
                chart_df = pd.concat(parts)
                fig = px.line(chart_df, x="Date", y="Index", color="Ticker")
                fig.update_traces(mode="lines", hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}")
                st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Comparison CSV",
            pd.DataFrame(results).to_csv(index=False).encode("utf-8"),
            file_name="Compare3_summary.csv",
            mime="text/csv"
        )

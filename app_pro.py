import io
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Naked Fundamentals PRO (Streamlit)", layout="wide")

# ============================
# Helpers & Models
# ============================
@dataclass
class Core:
    price: float=None; mktcap: float=None; sector: str=None; industry: str=None; beta: float=None; div_yield: float=None
    revenue: float=None; revenue_yoy: float=None; ebitda: float=None; ebit: float=None; ocf: float=None; capex: float=None; fcf: float=None
    total_assets: float=None; total_liab: float=None; equity: float=None; debt: float=None; cash_sti: float=None
    cur_assets: float=None; cur_liab: float=None; inventory: float=None; goodwill: float=None
    ev: float=None; pe: float=None; fpe: float=None; ps: float=None; pb: float=None; ev_ebitda: float=None
    de: float=None; int_cov: float=None; cur_ratio: float=None; quick_ratio: float=None; gw_assets: float=None
    fcf_yield: float=None; sbc_rev: float=None; roic: float=None
    ar_days: float=None; ap_days: float=None; inv_days: float=None; ccc: float=None
    cogs: float=None

def num(x):
    return None if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))) else x

@st.cache_data(ttl=3600)
def fetch_all(tk: str):
    t = yf.Ticker(tk)
    info = t.info or {}
    inc = getattr(t, "financials", pd.DataFrame())
    bal = getattr(t, "balance_sheet", pd.DataFrame())
    cfs = getattr(t, "cashflow", pd.DataFrame())
    try:
        hist = t.history(period="3y")
    except Exception:
        hist = pd.DataFrame()
    return info, inc, bal, cfs, hist

def pick_latest(df: pd.DataFrame, row: str):
    try:
        s = df.loc[row].dropna()
        if len(s)==0: return None
        return float(s.iloc[0])
    except Exception:
        return None

def two_vals(df: pd.DataFrame, row: str):
    try:
        s = df.loc[row].dropna()
        if len(s) < 2: return None, None
        return float(s.iloc[0]), float(s.iloc[1])
    except Exception:
        return None, None

def compute_core(info, inc, bal, cfs) -> Core:
    price = info.get("currentPrice"); mktcap = info.get("marketCap")
    sector = info.get("sector"); industry = info.get("industry")
    beta = info.get("beta"); div_yield = info.get("dividendYield")
    revenue = pick_latest(inc, "Total Revenue") or pick_latest(inc, "TotalRevenue")
    ebit = pick_latest(inc, "Ebit") or pick_latest(inc, "EBIT")
    ebitda = pick_latest(inc, "Ebitda") or pick_latest(inc, "EBITDA")
    if ebitda is None and ebit is not None:
        dep = pick_latest(inc, "Depreciation") or 0.0
        amo = pick_latest(inc, "Amortization") or 0.0
        ebitda = ebit + dep + amo
    ocf = pick_latest(cfs, "Total Cash From Operating Activities") or pick_latest(cfs, "Operating Cash Flow")
    capex = pick_latest(cfs, "Capital Expenditures") or pick_latest(cfs, "Investments In Property, Plant, And Equipment")
    if capex is not None and capex < 0: capex = -float(capex)
    fcf = (ocf - capex) if (ocf is not None and capex is not None) else None
    rev_t, rev_t1 = two_vals(inc, "Total Revenue")
    revenue_yoy = (rev_t/rev_t1 - 1.0) if (rev_t and rev_t1) else None

    total_assets = pick_latest(bal, "Total Assets")
    total_liab = pick_latest(bal, "Total Liab") or pick_latest(bal, "Total Liabilities Net Minority Interest")
    equity = pick_latest(bal, "Total Stockholder Equity")
    debt = (pick_latest(bal, "Short Long Term Debt") or 0.0) + (pick_latest(bal, "Long Term Debt") or 0.0)
    cash_sti = (pick_latest(bal, "Cash And Cash Equivalents") or 0.0) + (pick_latest(bal, "Short Term Investments") or 0.0)
    cur_assets = pick_latest(bal, "Total Current Assets")
    cur_liab = pick_latest(bal, "Total Current Liabilities")
    inventory = pick_latest(bal, "Inventory")
    goodwill = pick_latest(bal, "Goodwill")
    ar = pick_latest(bal, "Net Receivables") or pick_latest(bal, "Accounts Receivable")
    ap = pick_latest(bal, "Accounts Payable")
    cogs = pick_latest(inc, "Cost Of Revenue")

    ev = (mktcap + debt - cash_sti) if (mktcap is not None) else None
    pe = info.get("trailingPE"); fpe = info.get("forwardPE")
    ps = (mktcap / revenue) if (mktcap and revenue) else None
    pb = (mktcap / (equity)) if (mktcap and equity and equity!=0) else None
    ev_ebitda = (ev / ebitda) if (ev and ebitda and ebitda!=0) else None
    de = (debt / equity) if (debt and equity and equity!=0) else None
    int_exp = pick_latest(inc, "Interest Expense")
    int_cov = (ebit / abs(int_exp)) if (ebit and int_exp not in (None, 0, 0.0)) else None
    cur_ratio = (cur_assets / cur_liab) if (cur_assets and cur_liab and cur_liab!=0) else None
    quick_ratio = ((cur_assets - (inventory or 0.0)) / cur_liab) if (cur_assets and cur_liab and cur_liab!=0) else None
    gw_assets = (goodwill / total_assets) if (goodwill and total_assets and total_assets!=0) else None
    fcf_yield = (fcf / mktcap) if (fcf is not None and mktcap) else None
    sbc = pick_latest(cfs, "Stock Based Compensation")
    sbc_rev = (sbc / revenue) if (sbc is not None and revenue) else None

    # Working capital days
    AR_days = (ar / revenue)*365.0 if (ar is not None and revenue) else None
    AP_days = (ap / cogs)*365.0 if (ap is not None and cogs) else None
    INV_days = (inventory / cogs)*365.0 if (inventory is not None and cogs) else None
    ccc = AR_days + INV_days - AP_days if all(v is not None for v in [AR_days, AP_days, INV_days]) else None

    # ROIC approx
    tax = 0.21
    nopat = (ebit*(1-tax)) if (ebit is not None) else None
    invested_capital = None
    if total_assets is not None and cur_liab is not None and cash_sti is not None:
        invested_capital = total_assets - cur_liab - cash_sti
    roic = (nopat / invested_capital) if (nopat is not None and invested_capital and invested_capital!=0) else None

    return Core(
        price=num(price), mktcap=num(mktcap), sector=sector, industry=industry, beta=num(beta), div_yield=num(div_yield),
        revenue=num(revenue), revenue_yoy=num(revenue_yoy), ebitda=num(ebitda), ebit=num(ebit), ocf=num(ocf), capex=num(capex), fcf=num(fcf),
        total_assets=num(total_assets), total_liab=num(total_liab), equity=num(equity), debt=num(debt), cash_sti=num(cash_sti),
        cur_assets=num(cur_assets), cur_liab=num(cur_liab), inventory=num(inventory), goodwill=num(goodwill),
        ev=num(ev), pe=num(pe), fpe=num(fpe), ps=num(ps), pb=num(pb), ev_ebitda=num(ev_ebitda),
        de=num(de), int_cov=num(int_cov), cur_ratio=num(cur_ratio), quick_ratio=num(quick_ratio), gw_assets=num(gw_assets),
        fcf_yield=num(fcf_yield), sbc_rev=num(sbc_rev), roic=num(roic),
        ar_days=num(AR_days), ap_days=num(AP_days), inv_days=num(INV_days), ccc=num(ccc), cogs=num(cogs)
    )

def momentum_metrics(hist: pd.DataFrame):
    out = {}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out, pd.DataFrame()
    close = hist["Close"].dropna()
    out["ret_1m"] = close.pct_change(21).iloc[-1] if len(close)>=22 else None
    out["ret_3m"] = close.pct_change(63).iloc[-1] if len(close)>=64 else None
    out["ret_6m"] = close.pct_change(126).iloc[-1] if len(close)>=127 else None
    out["ret_12m"] = close.pct_change(252).iloc[-1] if len(close)>=253 else None
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["sma50"] = float(sma50.iloc[-1]) if len(sma50.dropna())>0 else None
    out["sma200"] = float(sma200.iloc[-1]) if len(sma200.dropna())>0 else None
    out["price"] = float(close.iloc[-1])
    out["hi_52w"] = float(close.tail(252).max()) if len(close)>=252 else None
    out["px_vs_52w_hi"] = (out["price"]/out["hi_52w"] - 1.0) if (out["price"] and out["hi_52w"]) else None
    df = pd.DataFrame({"Close": close, "SMA50": sma50, "SMA200": sma200}).reset_index()
    return out, df

def forensics(inc, bal, cfs, core: Core):
    # Altman Z
    TA=core.total_assets; TL=core.total_liab; CA=core.cur_assets; CL=core.cur_liab
    RE = pick_latest(bal, "Retained Earnings")
    EBIT = core.ebit; Sales=core.revenue; MV=core.mktcap
    WC = (CA-CL) if (CA is not None and CL is not None) else None
    z=None
    if all(v is not None for v in [TA, TL, WC, RE, EBIT, Sales, MV]) and TA!=0 and TL!=0:
        X1=WC/TA; X2=RE/TA; X3=EBIT/TA; X4=MV/TL; X5=Sales/TA
        z=1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # Piotroski F
    NI_t, NI_t1 = two_vals(inc, "Net Income")
    CFO_t, CFO_t1 = two_vals(cfs, "Total Cash From Operating Activities")
    if CFO_t is None: CFO_t, CFO_t1 = two_vals(cfs, "Operating Cash Flow")
    ROA_t = (NI_t/TA) if (NI_t is not None and TA) else None
    LTD_t = pick_latest(bal, "Long Term Debt")
    LTD_t1=None
    try:
        s=bal.loc["Long Term Debt"].dropna()
        if len(s)>=2: LTD_t1=float(s.iloc[1])
    except Exception: pass
    CR_t = (CA/CL) if (CA and CL) else None
    GM_t=GM_t1=None
    try:
        gp = inc.loc["Gross Profit"].dropna(); rev = inc.loc["Total Revenue"].dropna()
        if len(gp)>=2 and len(rev)>=2 and rev.iloc[0]!=0 and rev.iloc[1]!=0:
            GM_t=float(gp.iloc[0])/float(rev.iloc[0]); GM_t1=float(gp.iloc[1])/float(rev.iloc[1])
    except Exception: pass
    AT_t=AT_t1=None
    try:
        assets=bal.loc["Total Assets"].dropna(); rev=inc.loc["Total Revenue"].dropna()
        if len(assets)>=2 and len(rev)>=2 and assets.iloc[0]!=0 and assets.iloc[1]!=0:
            AT_t=float(rev.iloc[0])/float(assets.iloc[0]); AT_t1=float(rev.iloc[1])/float(assets.iloc[1])
    except Exception: pass
    fscore=0
    if NI_t is not None and NI_t>0: fscore+=1
    if CFO_t is not None and CFO_t>0: fscore+=1
    if ROA_t is not None and ROA_t>0: fscore+=1
    if (CFO_t is not None and NI_t is not None) and (CFO_t>NI_t): fscore+=1
    if LTD_t is not None and LTD_t1 is not None and LTD_t<=LTD_t1: fscore+=1
    if CR_t is not None and CR_t>=1.0: fscore+=1
    if (GM_t is not None and GM_t1 is not None) and (GM_t>=GM_t1): fscore+=1
    if (AT_t is not None and AT_t1 is not None) and (AT_t>=AT_t1): fscore+=1

    # Accruals and DSO
    NI = pick_latest(inc, "Net Income")
    OCF = core.ocf
    accr = (NI - OCF) / TA if (NI is not None and OCF is not None and TA) else None
    dso = core.ar_days
    return {"AltmanZ": num(z), "PiotroskiF": num(fscore), "Accruals": num(accr), "DSO": num(dso)}

def value_quality_momentum(core: Core, fx: Dict, mom: Dict):
    # Value
    pe = core.pe; ev_e = core.ev_ebitda; fcfy = core.fcf_yield
    pe_score = 60 if pe is None else max(0, min(100, 100 - ((pe-10)*5))) if pe>=10 else 100
    ev_score = 100 if ev_e is None else max(0, min(100, 100 - ((ev_e-8)*8)))
    fcf_score = 50 if fcfy is None else max(0, min(100, fcfy*1000))  # 10% -> 100
    value_comp = 0.34*pe_score + 0.33*ev_score + 0.33*fcf_score

    # Quality
    roic = core.roic or 0
    accr = fx.get("Accruals")
    accr_score = 100 if accr is None else max(0, 100 - (accr*400))
    fscore = fx.get("PiotroskiF") or 5; fscore_scaled = (fscore/9)*100
    quality_comp = 0.4*min(100, roic*100/20) + 0.3*accr_score + 0.3*fscore_scaled

    # Momentum
    r12 = mom.get("ret_12m")
    sma200 = mom.get("sma200"); price = mom.get("price")
    above200 = (price > sma200) if (price and sma200) else None
    r12_score = 50 if r12 is None else max(0, min(100, (r12*400)))
    trend_bonus = 15 if (above200 is True) else (-10 if above200 is False else 0)
    mom_score = max(0, min(100, r12_score + trend_bonus))

    total = 0.4*value_comp + 0.35*quality_comp + 0.25*mom_score
    return total, {"value": value_comp, "quality": quality_comp, "momentum": mom_score}

def recommendation(score, buy_thr=70, sell_thr=35):
    if score>=buy_thr: return "BUY"
    if score<=sell_thr: return "SELL"
    return "HOLD"

def dcf_equity_value(base_fcf, wacc, g, years_high=5, high_growth=0.08, net_debt=0.0):
    if base_fcf is None or wacc is None: return None
    fcfs = [base_fcf * ((1+high_growth)**i) for i in range(1, years_high+1)]
    pv = sum(fcfs[i-1]/((1+wacc)**i) for i in range(1, years_high+1))
    if wacc is None or g is None or wacc<=g: return None
    terminal = (fcfs[-1]*(1+g))/(wacc-g) if fcfs else None
    if terminal is None: return None
    pv_terminal = terminal/((1+wacc)**years_high)
    ev = pv + pv_terminal
    equity = ev - (net_debt or 0.0)
    return equity

def reverse_dcf_implied_growth(base_fcf, wacc, g, years_high, target_equity, net_debt=0.0):
    if any(v in (None, 0) for v in [base_fcf, wacc]) or g is None or target_equity is None:
        return None
    lo, hi = 0.0, 0.30
    for _ in range(40):
        mid = (lo+hi)/2
        eq = dcf_equity_value(base_fcf, wacc, g, years_high, mid, net_debt)
        if eq is None: return None
        if eq < target_equity: lo = mid
        else: hi = mid
    return (lo+hi)/2

def capital_allocation(cfs, core: Core):
    div = pick_latest(cfs, "Dividends Paid")
    buy = pick_latest(cfs, "Repurchase Of Stock")
    sbc = pick_latest(cfs, "Stock Based Compensation")
    payout = (abs(div)/core.ocf) if (div is not None and core.ocf) else None
    buy_yield = (abs(buy)/core.mktcap) if (buy is not None and core.mktcap) else None
    return {"Dividends": num(div), "Buybacks": num(buy), "SBC": num(sbc), "Payout/OCF": num(payout), "Buyback Yield": num(buy_yield), "SBC%": core.sbc_rev}

def build_summary_table(tk, core: Core, fx: Dict, mom: Dict, score, parts, verdict):
    data = {
        "Ticker": tk, "Price": core.price, "MktCap": core.mktcap, "EV": core.ev,
        "P/E": core.pe, "EV/EBITDA": core.ev_ebitda, "P/S": core.ps, "P/B": core.pb,
        "Debt/Equity": core.de, "Interest Coverage": core.int_cov, "Current Ratio": core.cur_ratio, "Quick Ratio": core.quick_ratio,
        "Revenue": core.revenue, "Revenue YoY": core.revenue_yoy, "EBITDA": core.ebitda, "EBIT": core.ebit, "OCF": core.ocf, "CAPEX": core.capex, "FCF": core.fcf,
        "FCF Yield": core.fcf_yield, "ROIC": core.roic, "Goodwill/Assets": core.gw_assets, "SBC%": core.sbc_rev,
        "Altman Z": fx.get("AltmanZ"), "Piotroski F": fx.get("PiotroskiF"), "Accruals": fx.get("Accruals"), "DSO(days)": fx.get("DSO"),
        "AR Days": core.ar_days, "AP Days": core.ap_days, "Inventory Days": core.inv_days, "CCC": core.ccc,
        "12M Return": mom.get("ret_12m"), "SMA200": mom.get("sma200"),
        "Score (0-100)": score, "V/Q/M": f"{parts['value']:.0f}/{parts['quality']:.0f}/{parts['momentum']:.0f}", "Recommendation": verdict
    }
    df = pd.DataFrame({"Metric": list(data.keys()), "Value": list(data.values())})
    return df

def format_pct_or_num(x):
    if x is None: return "—"
    try:
        if abs(x) < 1 and x not in (0, 1) and not float(x).is_integer():
            return f"{x:.2%}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)

# ============================
# UI
# ============================
st.title("🧠 Naked Fundamentals PRO — Streamlit")
st.caption("Forensic-grade fundamentals with Value/Quality/Momentum score, DCF, peer benchmarking, and BUY/HOLD/SELL signal.")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Single Ticker", "Compare 3"])
    if mode == "Single Ticker":
        tk1 = st.text_input("Ticker", "AEP").upper().strip()
    else:
        tk1 = st.text_input("Ticker 1", "AEP").upper().strip()
        tk2 = st.text_input("Ticker 2", "NEE").upper().strip()
        tk3 = st.text_input("Ticker 3", "FSLR").upper().strip()

    st.markdown("---")
    st.subheader("Valuation Settings")
    rf = st.number_input("Risk-Free Rate", value=0.045, step=0.005, format="%.3f")
    erp = st.number_input("Equity Risk Premium", value=0.050, step=0.005, format="%.3f")
    wacc_override = st.number_input("WACC Override (optional)", value=0.0, step=0.005, format="%.3f")
    term_g = st.number_input("Terminal Growth", value=0.02, step=0.005, format="%.3f")
    high_growth = st.number_input("High-Growth Years CAGR", value=0.08, step=0.01, format="%.3f")
    st.markdown("---")
    st.subheader("Scoring Thresholds")
    buy_thr = st.slider("BUY threshold", 50, 90, 70)
    sell_thr = st.slider("SELL threshold", 10, 50, 35)

    st.markdown("---")
    peer_str = st.text_input("Peers (comma-separated tickers, optional)", "")
    run = st.button("Run Analysis", type="primary")

if not run:
    st.info("Enter tickers and click **Run Analysis** from the left panel.")
    st.stop()

def analyze_one(tk: str) -> Tuple[Core, Dict, Dict, pd.DataFrame, float, Dict, str, Dict]:
    info, inc, bal, cfs, hist = fetch_all(tk)
    core = compute_core(info, inc, bal, cfs)
    mom, price_df = momentum_metrics(hist)
    fx = forensics(inc, bal, cfs, core)
    score, parts = value_quality_momentum(core, fx, mom)
    verdict = recommendation(score, buy_thr, sell_thr)
    cap = capital_allocation(cfs, core)
    return core, fx, mom, price_df, score, parts, verdict, cap

def peer_medians(peers: List[str]) -> Dict[str,float]:
    out={}
    vals = {k: [] for k in ["P/E","EV/EBITDA","P/S","P/B","Debt/Equity","FCF Yield","ROIC"]}
    for p in peers:
        try:
            i, inc, bal, cfs, _ = fetch_all(p)
            c = compute_core(i, inc, bal, cfs)
            if c.pe is not None: vals["P/E"].append(c.pe)
            if c.ev_ebitda is not None: vals["EV/EBITDA"].append(c.ev_ebitda)
            if c.ps is not None: vals["P/S"].append(c.ps)
            if c.pb is not None: vals["P/B"].append(c.pb)
            if c.de is not None: vals["Debt/Equity"].append(c.de)
            if c.fcf_yield is not None: vals["FCF Yield"].append(c.fcf_yield)
            if c.roic is not None: vals["ROIC"].append(c.roic)
        except Exception:
            continue
    for k,v in vals.items():
        if v:
            out[k]=float(np.median(v))
    return out

if mode == "Single Ticker":
    core, fx, mom, price_df, score, parts, verdict, cap = analyze_one(tk1)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ticker", tk1)
    c2.metric("Score", f"{score:.0f}")
    c3.metric("Verdict", verdict)
    c4.metric("Price", f"{core.price}" if core.price else "—")

    # Price chart
    if not price_df.empty:
        fig = px.line(price_df, x=price_df.columns[0], y=["Close","SMA50","SMA200"], title=f"{tk1} — Price (3y) & SMAs")
        st.plotly_chart(fig, use_container_width=True)

    # Summary & Forensics
    st.subheader("Summary / Forensics")
    sum_df = build_summary_table(tk1, core, fx, mom, score, parts, verdict)
    st.dataframe(sum_df, use_container_width=True, height=600)

    # Capital Allocation
    st.subheader("Capital Allocation")
    cap_df = pd.DataFrame({"Metric": list(cap.keys()), "Value": list(cap.values())})
    st.dataframe(cap_df, use_container_width=True)

    # Peer Benchmarking
    if peer_str.strip():
        peers = [p.strip().upper() for p in peer_str.split(",") if p.strip()]
        meds = peer_medians(peers)
        if meds:
            bench_rows = []
            for m in ["P/E","EV/EBITDA","P/S","P/B","Debt/Equity","FCF Yield","ROIC"]:
                bench_rows.append({"Metric": m, "Company": getattr(core, "pe" if m=="P/E" else
                                                                   "ev_ebitda" if m=="EV/EBITDA" else
                                                                   "ps" if m=="P/S" else
                                                                   "pb" if m=="P/B" else
                                                                   "de" if m=="Debt/Equity" else
                                                                   "fcf_yield" if m=="FCF Yield" else
                                                                   "roic"),
                                   "Peer Median": meds.get(m)})
            bench_df = pd.DataFrame(bench_rows)
            st.subheader("Peer Benchmark (Medians)")
            st.dataframe(bench_df, use_container_width=True)

    # DCF & Reverse DCF
    st.subheader("DCF / Reverse DCF")
    net_debt = (core.debt or 0) - (core.cash_sti or 0)
    wacc = (wacc_override if wacc_override>0 else (rf+erp))
    eq_val = dcf_equity_value(core.fcf, wacc, term_g, years_high=5, high_growth=high_growth, net_debt=net_debt)
    imp_g = reverse_dcf_implied_growth(core.fcf, wacc, term_g, 5, core.mktcap, net_debt=net_debt)
    dcf_table = pd.DataFrame({
        "Item": ["Base FCF","WACC","Terminal g","High-growth CAGR (5y)","DCF Equity Value","Implied Upside","Reverse DCF: Implied 5y CAGR"],
        "Value": [core.fcf, wacc, term_g, high_growth, eq_val,
                  (eq_val/core.mktcap - 1.0) if (eq_val and core.mktcap) else None,
                  imp_g]
    })
    st.dataframe(dcf_table, use_container_width=True)

    # EV/EBITDA Sensitivity
    st.subheader("EV/EBITDA Sensitivity (Equity)")
    mults = [8,10,12,14,16,18]
    rows=[]
    ebitda = core.ebitda or 0
    for m in mults:
        ev = m*ebitda if ebitda else None
        eq = (ev - net_debt) if ev is not None else None
        rows.append({"EV/EBITDA": m, "EBITDA": core.ebitda, "NetDebt": net_debt, "Implied Equity": eq})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Downloads (Excel)
    with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as writer:
        sum_df.to_excel(writer, index=False, sheet_name="Summary")
        cap_df.to_excel(writer, index=False, sheet_name="CapitalAlloc")
        dcf_table.to_excel(writer, index=False, sheet_name="DCF")
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Sensitivity")
        if not price_df.empty:
            price_df.to_excel(writer, index=False, sheet_name="Prices")
        data = writer._save() or writer._handles.handle.getvalue()
    st.download_button("Download Excel", data, file_name=f"{tk1}_Naked_PRO.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    results = []
    charts = []
    for tk in [tk1, tk2, tk3]:
        try:
            core, fx, mom, price_df, score, parts, verdict, cap = analyze_one(tk)
        except Exception as e:
            st.error(f"Failed to analyze {tk}: {e}")
            continue
        results.append({
            "Ticker": tk, "Score": round(score,1), "Verdict": verdict, "Price": core.price, "MktCap": core.mktcap,
            "P/E": core.pe, "EV/EBITDA": core.ev_ebitda, "D/E": core.de, "FCF Yield": core.fcf_yield, "12M Ret": mom.get("ret_12m")
        })
        if not price_df.empty:
            charts.append((tk, price_df))

    if not results:
        st.error("No valid tickers to compare.")
        st.stop()

    cmp_df = pd.DataFrame(results)
    st.subheader("Comparison Summary")
    st.dataframe(cmp_df, use_container_width=True)

    # Normalized price plot
    if charts:
        st.subheader("Normalized Prices (last = 100)")
        parts=[]
        for tk, dfp in charts:
            ser = dfp["Close"].dropna()
            if ser.empty: continue
            base = ser.iloc[-1]
            parts.append(pd.DataFrame({"Date": dfp.iloc[:,0], "Ticker": tk, "Index": ser/base*100}))
        if parts:
            chart_df = pd.concat(parts)
            fig = px.line(chart_df, x="Date", y="Index", color="Ticker")
            st.plotly_chart(fig, use_container_width=True)

    # Download comparison
    st.download_button("Download Comparison CSV", cmp_df.to_csv(index=False).encode("utf-8"), file_name="Compare3_summary.csv", mime="text/csv")

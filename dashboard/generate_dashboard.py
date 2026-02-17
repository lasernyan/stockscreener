#!/usr/bin/env python3
"""
Gold & Silver Dashboard Generator
Reads JSON data and produces a self-contained dark-themed HTML dashboard.
"""

import json
import os
import sys
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = SCRIPT_DIR


def load_data():
    path = os.path.join(DATA_DIR, "dashboard_latest.json")
    if not os.path.exists(path):
        print(f"[ERROR] Data file not found: {path}", file=sys.stderr)
        print("  Run fetch_data.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def fmt_num(val, prefix="", suffix="", decimals=2):
    """Format a number for display."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    if abs(val) >= 1_000_000:
        return f"{prefix}{val/1_000_000:,.{decimals}f}M{suffix}"
    if abs(val) >= 1_000:
        return f"{prefix}{val:,.{decimals}f}{suffix}"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def change_class(val):
    if val is None:
        return "neutral"
    return "positive" if val > 0 else "negative" if val < 0 else "neutral"


def change_arrow(val):
    if val is None:
        return ""
    return "&#9650;" if val > 0 else "&#9660;" if val < 0 else "&#9644;"


def signal_dot(signal):
    colors = {"bullish": "#00e676", "bearish": "#ff1744", "neutral": "#ffc107"}
    c = colors.get(signal, colors["neutral"])
    return f'<span class="dot" style="background:{c}"></span>'


def generate_html(data):
    d = data
    p = d["prices"]
    m = d["macro"]
    e = d["etf_flows"]
    c = d["cot"]
    cb = d["central_banks"]
    ch = d["china"]
    ss = d["silver_supply"]
    tp = d["trading_plan"]
    mg = d["margin"]
    sig = d["signals"]["triple_factor"]
    prev = d.get("previous", {})

    gold = p["gold"]
    silver = p["silver"]

    gold_chg_cls = change_class(gold["change_pct"])
    silver_chg_cls = change_class(silver["change_pct"])
    gold_arrow = change_arrow(gold["change_pct"])
    silver_arrow = change_arrow(silver["change_pct"])

    # Previous day comparison
    prev_gold_note = ""
    if prev.get("gold_price") and prev.get("date"):
        diff = gold["price"] - prev["gold_price"]
        prev_gold_note = f'<span class="prev-compare">vs {prev["date"]}: {diff:+,.0f}</span>'

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gold &amp; Silver Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {{
    --bg-primary: #0a0a0f;
    --bg-card: #12121a;
    --bg-card-hover: #1a1a25;
    --bg-elevated: #1e1e2a;
    --border: #2a2a3a;
    --border-glow: rgba(100, 100, 255, 0.1);
    --text-primary: #e8e8f0;
    --text-secondary: #8888a0;
    --text-muted: #555570;
    --gold: #f5c842;
    --gold-dim: #b8962f;
    --gold-glow: rgba(245, 200, 66, 0.15);
    --silver: #c0c0d0;
    --silver-dim: #8080a0;
    --silver-glow: rgba(192, 192, 208, 0.12);
    --green: #00e676;
    --green-dim: rgba(0, 230, 118, 0.15);
    --red: #ff1744;
    --red-dim: rgba(255, 23, 68, 0.15);
    --blue: #448aff;
    --blue-dim: rgba(68, 138, 255, 0.15);
    --purple: #b388ff;
    --cyan: #18ffff;
    --orange: #ff9100;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.5;
  }}

  .dashboard {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 24px;
  }}

  /* Header */
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.5px;
  }}
  .header h1 span.gold-text {{ color: var(--gold); }}
  .header h1 span.silver-text {{ color: var(--silver); }}
  .header .meta {{
    text-align: right;
    font-size: 12px;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
  }}
  .header .meta .live-dot {{
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 4px;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
  }}

  /* Grid */
  .grid {{ display: grid; gap: 16px; }}
  .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
  .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}

  @media (max-width: 1200px) {{
    .grid-4 {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-3 {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  @media (max-width: 768px) {{
    .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
    .dashboard {{ padding: 12px; }}
  }}

  /* Cards */
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }}
  .card:hover {{
    background: var(--bg-card-hover);
    border-color: rgba(100, 100, 255, 0.2);
  }}
  .card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: transparent;
  }}
  .card.gold-accent::before {{ background: linear-gradient(90deg, var(--gold), transparent); }}
  .card.silver-accent::before {{ background: linear-gradient(90deg, var(--silver), transparent); }}
  .card.blue-accent::before {{ background: linear-gradient(90deg, var(--blue), transparent); }}
  .card.green-accent::before {{ background: linear-gradient(90deg, var(--green), transparent); }}
  .card.purple-accent::before {{ background: linear-gradient(90deg, var(--purple), transparent); }}
  .card.red-accent::before {{ background: linear-gradient(90deg, var(--red), transparent); }}
  .card.cyan-accent::before {{ background: linear-gradient(90deg, var(--cyan), transparent); }}

  .card-title {{
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .card-title .icon {{ font-size: 14px; }}

  /* Price cards */
  .price-main {{
    font-size: 36px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -1px;
    margin-bottom: 4px;
  }}
  .price-main.gold-price {{ color: var(--gold); }}
  .price-main.silver-price {{ color: var(--silver); }}

  .price-change {{
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
  }}
  .price-change.positive {{ color: var(--green); background: var(--green-dim); }}
  .price-change.negative {{ color: var(--red); background: var(--red-dim); }}
  .price-change.neutral {{ color: var(--text-secondary); }}

  .price-detail {{
    margin-top: 12px;
    font-size: 12px;
    color: var(--text-secondary);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
  }}
  .price-detail span {{ font-family: 'JetBrains Mono', monospace; }}

  .prev-compare {{
    font-size: 11px;
    color: var(--text-muted);
    margin-left: 8px;
  }}

  /* Metric rows */
  .metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 13px;
  }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-label {{ color: var(--text-secondary); }}
  .metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    font-size: 13px;
  }}
  .metric-value.positive {{ color: var(--green); }}
  .metric-value.negative {{ color: var(--red); }}
  .metric-value.gold-color {{ color: var(--gold); }}
  .metric-value.silver-color {{ color: var(--silver); }}
  .metric-value.blue-color {{ color: var(--blue); }}
  .metric-value.purple-color {{ color: var(--purple); }}

  /* Signal dots */
  .dot {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 4px;
  }}

  /* Score card */
  .score-card {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 120px;
  }}
  .score-value {{
    font-size: 48px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, var(--gold), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .score-label {{
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
  }}

  /* Section titles */
  .section-title {{
    font-size: 15px;
    font-weight: 600;
    margin: 28px 0 14px 0;
    padding-left: 12px;
    border-left: 3px solid var(--blue);
    color: var(--text-primary);
    letter-spacing: -0.3px;
  }}

  /* Trade plan */
  .trade-entry {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
  }}
  .trade-label {{
    color: var(--text-muted);
    text-align: right;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding-top: 1px;
  }}
  .trade-val {{ font-weight: 500; }}
  .trade-val.tp {{ color: var(--green); }}
  .trade-val.sl {{ color: var(--red); }}
  .trade-val.entry {{ color: var(--blue); }}

  /* Bar chart */
  .bar-chart {{ margin-top: 8px; }}
  .bar-row {{
    display: flex;
    align-items: center;
    margin-bottom: 6px;
    font-size: 11px;
  }}
  .bar-label {{
    width: 50px;
    color: var(--text-secondary);
    text-align: right;
    padding-right: 8px;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
  }}
  .bar-track {{
    flex: 1;
    height: 16px;
    background: rgba(255,255,255,0.03);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
    display: flex;
    align-items: center;
    padding-left: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    color: rgba(255,255,255,0.9);
    min-width: fit-content;
  }}
  .bar-fill.positive {{ background: linear-gradient(90deg, rgba(0,230,118,0.6), rgba(0,230,118,0.3)); }}
  .bar-fill.negative {{ background: linear-gradient(90deg, rgba(255,23,68,0.6), rgba(255,23,68,0.3)); }}

  /* Fullwidth cards */
  .full-width {{ grid-column: 1 / -1; }}

  /* Tag */
  .tag {{
    display: inline-block;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .tag.long {{ background: var(--green-dim); color: var(--green); }}
  .tag.short {{ background: var(--red-dim); color: var(--red); }}
  .tag.main {{ background: var(--blue-dim); color: var(--blue); }}
  .tag.sub {{ background: rgba(255,255,255,0.05); color: var(--text-secondary); }}
  .tag.warn {{ background: rgba(255,145,0,0.15); color: var(--orange); }}

  /* Footer */
  .footer {{
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-muted);
    text-align: center;
    line-height: 1.8;
  }}

  /* Ratio gauge */
  .gauge-container {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 8px;
  }}
  .gauge-bar {{
    flex: 1;
    height: 8px;
    background: linear-gradient(90deg, var(--green) 0%, var(--gold) 50%, var(--red) 100%);
    border-radius: 4px;
    position: relative;
  }}
  .gauge-marker {{
    position: absolute;
    top: -4px;
    width: 3px;
    height: 16px;
    background: white;
    border-radius: 2px;
    box-shadow: 0 0 6px rgba(255,255,255,0.5);
  }}
  .gauge-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
  }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>
<div class="dashboard">

  <!-- Header -->
  <div class="header">
    <h1><span class="gold-text">Gold</span> &amp; <span class="silver-text">Silver</span> Trading Dashboard</h1>
    <div class="meta">
      <div><span class="live-dot"></span> {d['generated_date']} {d['generated_time']}</div>
      <div style="margin-top:2px;color:var(--text-muted)">Auto-updated daily</div>
    </div>
  </div>

  <!-- Row 1: Price Cards + Ratio + Triple Factor -->
  <div class="grid grid-4">

    <!-- Gold Price -->
    <div class="card gold-accent">
      <div class="card-title"><span class="icon">Au</span> XAU/USD</div>
      <div class="price-main gold-price">${gold['price']:,.0f}</div>
      <span class="price-change {gold_chg_cls}">{gold_arrow} {gold['change_pct']:+.2f}%</span>
      {prev_gold_note}
      <div class="price-detail">
        <span>RSI: {gold['rsi']}</span>
        <span>50SMA: ${gold['sma_50']:,.0f}</span>
        <span>52W H: ${gold['high_52w']:,.0f}</span>
        <span>52W L: ${gold['low_52w']:,.0f}</span>
      </div>
    </div>

    <!-- Silver Price -->
    <div class="card silver-accent">
      <div class="card-title"><span class="icon">Ag</span> XAG/USD</div>
      <div class="price-main silver-price">${silver['price']:,.2f}</div>
      <span class="price-change {silver_chg_cls}">{silver_arrow} {silver['change_pct']:+.2f}%</span>
      <div class="price-detail">
        <span>ATH: ${silver['ath']:,.2f}</span>
        <span>from ATH: {((silver['price']/silver['ath'])-1)*100:+.1f}%</span>
        <span>52W H: ${silver['high_52w']:,.2f}</span>
        <span>52W L: ${silver['low_52w']:,.2f}</span>
      </div>
    </div>

    <!-- Gold/Silver Ratio -->
    <div class="card cyan-accent">
      <div class="card-title"><span class="icon">Au/Ag</span> Gold/Silver Ratio</div>
      <div class="price-main" style="font-size:32px;color:var(--cyan)">{p['ratio']}:1</div>
      <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">Historical avg: {p['ratio_historical_avg']}:1</div>
      <div class="gauge-container">
        <div style="font-size:10px;color:var(--text-muted);width:24px">40</div>
        <div style="flex:1">
          <div class="gauge-bar">
            <div class="gauge-marker" style="left:{max(0,min(100, (p['ratio']-40)/(100-40)*100))}%"></div>
          </div>
          <div class="gauge-labels">
            <span>Ag Cheap</span>
            <span>60-70</span>
            <span>Ag Rich</span>
          </div>
        </div>
        <div style="font-size:10px;color:var(--text-muted);width:28px">100</div>
      </div>
    </div>

    <!-- Triple Factor Score -->
    <div class="card green-accent">
      <div class="card-title"><span class="icon">&#9733;</span> Triple Factor Signal</div>
      <div class="score-card">
        <div class="score-value">{sig['score']}</div>
        <div class="score-label">BULLISH</div>
      </div>
      <div style="margin-top:8px;font-size:12px">
        <div class="metric-row">
          <span class="metric-label">{signal_dot(sig['real_yield'])} Real Yield</span>
          <span class="metric-value positive">{m['tips_10y']}% (declining)</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">{signal_dot(sig['dxy'])} DXY</span>
          <span class="metric-value positive">{m['dxy']} ({m['dxy_change_1m_pct']:+.2f}% 1M)</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">{signal_dot(sig['central_bank'])} CB Demand</span>
          <span class="metric-value positive">{cb['annual_demand_tonnes']}t/yr</span>
        </div>
      </div>
    </div>

  </div>

  <!-- Row 2: Macro Factors -->
  <div class="section-title">Macro &amp; Real Yield Analysis</div>
  <div class="grid grid-3">

    <!-- TIPS / Real Yield -->
    <div class="card blue-accent">
      <div class="card-title"><span class="icon">&#x25CE;</span> US Real Yield (TIPS)</div>
      <div style="text-align:center;margin:8px 0">
        <div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono';color:var(--blue)">{m['tips_10y']}%</div>
        <div style="font-size:11px;color:var(--text-muted)">10Y TIPS Real Yield</div>
      </div>
      <div style="font-size:11px;color:var(--text-muted);margin:8px 0;padding:8px;background:var(--bg-elevated);border-radius:6px;font-family:'JetBrains Mono'">
        Real = Nominal - BEI<br>
        {m['tips_10y']}% = {m['nominal_10y']}% - {m['bei_10y']}%
      </div>
      <div class="metric-row">
        <span class="metric-label">Nominal 10Y</span>
        <span class="metric-value">{m['nominal_10y']}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">BEI (Exp. Inflation)</span>
        <span class="metric-value">{m['bei_10y']}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">FF Rate</span>
        <span class="metric-value">{m['ff_rate']}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">March Cut Prob</span>
        <span class="metric-value">{m['ff_rate_cut_prob_march']}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Latest CPI</span>
        <span class="metric-value">{m['cpi_latest']}%</span>
      </div>
    </div>

    <!-- DXY -->
    <div class="card purple-accent">
      <div class="card-title"><span class="icon">$</span> DXY (Dollar Index)</div>
      <div style="text-align:center;margin:8px 0">
        <div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono';color:var(--purple)">{m['dxy']}</div>
        <div style="font-size:11px;color:var(--text-muted)">Correlation: -0.75 vs Gold</div>
      </div>
      <div class="metric-row">
        <span class="metric-label">1M Change</span>
        <span class="metric-value negative">{m['dxy_change_1m_pct']:+.2f}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">12M Change</span>
        <span class="metric-value negative">{m['dxy_change_12m_pct']:+.2f}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">USD Reserve Share</span>
        <span class="metric-value negative">56% (30Y low)</span>
      </div>
      <div style="margin-top:10px;padding:8px;background:var(--bg-elevated);border-radius:6px;font-size:11px;color:var(--text-muted)">
        DXY&#8595; = Gold&#8593; (inverse correlation)<br>
        De-dollarization trend continues
      </div>
    </div>

    <!-- CME Margins -->
    <div class="card red-accent">
      <div class="card-title"><span class="icon">&#9888;</span> CME Margin Requirements</div>
      <div style="display:flex;gap:12px;margin:12px 0">
        <div style="flex:1;text-align:center;padding:12px;background:var(--gold-glow);border-radius:8px">
          <div style="font-size:24px;font-weight:700;font-family:'JetBrains Mono';color:var(--gold)">{mg['gold_pct']}%</div>
          <div style="font-size:10px;color:var(--text-muted);margin-top:2px">Gold (GC)</div>
        </div>
        <div style="flex:1;text-align:center;padding:12px;background:var(--silver-glow);border-radius:8px">
          <div style="font-size:24px;font-weight:700;font-family:'JetBrains Mono';color:var(--silver)">{mg['silver_pct']}%</div>
          <div style="font-size:10px;color:var(--text-muted);margin-top:2px">Silver (SI)</div>
        </div>
      </div>
      <div style="padding:8px;background:var(--red-dim);border-radius:6px;font-size:11px;color:var(--red)">
        Silver margin is 2x Gold<br>
        SPAN&#8594;%base shift since Jan 13<br>
        3 hikes since transition
      </div>
      <div class="metric-row">
        <span class="metric-label">Gold per contract</span>
        <span class="metric-value">~$45,387</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Silver per contract</span>
        <span class="metric-value">~$68,958</span>
      </div>
    </div>

  </div>

  <!-- Row 3: Fund Flows -->
  <div class="section-title">Fund Flows &amp; Positioning</div>
  <div class="grid grid-3">

    <!-- GLD Flows -->
    <div class="card gold-accent">
      <div class="card-title"><span class="icon">&#9670;</span> GLD ETF Flows</div>
      <div class="metric-row">
        <span class="metric-label">AUM</span>
        <span class="metric-value gold-color">${e['GLD']['aum_b']}B</span>
      </div>
      <div class="bar-chart">
        <div class="bar-row">
          <span class="bar-label">5D</span>
          <div class="bar-track">
            <div class="bar-fill negative" style="width:{min(abs(e['GLD']['flow_5d_m'])/300*100, 100):.0f}%">{e['GLD']['flow_5d_m']:+.0f}M</div>
          </div>
        </div>
        <div class="bar-row">
          <span class="bar-label">1M</span>
          <div class="bar-track">
            <div class="bar-fill positive" style="width:{min(e['GLD']['flow_1m_b']/5*100, 100):.0f}%">+${e['GLD']['flow_1m_b']}B</div>
          </div>
        </div>
        <div class="bar-row">
          <span class="bar-label">3M</span>
          <div class="bar-track">
            <div class="bar-fill positive" style="width:{min(e['GLD']['flow_3m_b']/10*100, 100):.0f}%">+${e['GLD']['flow_3m_b']}B</div>
          </div>
        </div>
        <div class="bar-row">
          <span class="bar-label">1Y</span>
          <div class="bar-track">
            <div class="bar-fill positive" style="width:{min(e['GLD']['flow_1y_b']/30*100, 100):.0f}%">+${e['GLD']['flow_1y_b']}B</div>
          </div>
        </div>
      </div>
    </div>

    <!-- SLV Flows -->
    <div class="card silver-accent">
      <div class="card-title"><span class="icon">&#9670;</span> SLV ETF Flows</div>
      <div class="metric-row">
        <span class="metric-label">AUM</span>
        <span class="metric-value silver-color">${e['SLV']['aum_b']}B</span>
      </div>
      <div class="bar-chart">
        <div class="bar-row">
          <span class="bar-label">5D</span>
          <div class="bar-track">
            <div class="bar-fill negative" style="width:{min(abs(e['SLV']['flow_5d_m'])/500*100, 100):.0f}%">{e['SLV']['flow_5d_m']:+.0f}M</div>
          </div>
        </div>
        <div class="bar-row">
          <span class="bar-label">1M</span>
          <div class="bar-track">
            <div class="bar-fill negative" style="width:{min(abs(e['SLV']['flow_1m_m'])/500*100, 100):.0f}%">{e['SLV']['flow_1m_m']:+.0f}M</div>
          </div>
        </div>
        <div class="bar-row">
          <span class="bar-label">3M</span>
          <div class="bar-track">
            <div class="bar-fill positive" style="width:{min(e['SLV']['flow_3m_b']/5*100, 100):.0f}%">+${e['SLV']['flow_3m_b']}B</div>
          </div>
        </div>
        <div class="bar-row">
          <span class="bar-label">1Y</span>
          <div class="bar-track">
            <div class="bar-fill positive" style="width:{min(e['SLV']['flow_1y_b']/5*100, 100):.0f}%">+${e['SLV']['flow_1y_b']}B</div>
          </div>
        </div>
      </div>
    </div>

    <!-- COT Positioning -->
    <div class="card blue-accent">
      <div class="card-title"><span class="icon">&#9776;</span> COT Positioning (COMEX)</div>
      <div style="font-size:11px;font-weight:600;color:var(--gold);margin-bottom:6px">GOLD</div>
      <div class="metric-row">
        <span class="metric-label">Spec Net Long</span>
        <span class="metric-value positive">{c['gold']['spec_net_long']:,}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Managed Money</span>
        <span class="metric-value positive">{c['gold']['managed_net_long']:,}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Total OI</span>
        <span class="metric-value">{c['gold']['total_oi']:,}</span>
      </div>
      <div style="font-size:11px;font-weight:600;color:var(--silver);margin:10px 0 6px">SILVER</div>
      <div class="metric-row">
        <span class="metric-label">Backwardation</span>
        <span class="metric-value {'positive' if c['silver']['backwardation'] else 'negative'}">{'Active' if c['silver']['backwardation'] else 'No'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Shanghai Premium</span>
        <span class="metric-value positive">+{c['silver']['shanghai_premium_pct']}%</span>
      </div>
      <div style="margin-top:8px;padding:6px;background:var(--bg-elevated);border-radius:4px;font-size:10px;color:var(--text-muted)">
        Spec longs NOT stretched at ATH = room to add
      </div>
    </div>

  </div>

  <!-- Row 4: Central Banks & China -->
  <div class="section-title">Central Banks &amp; China Flows</div>
  <div class="grid grid-2">

    <!-- Central Banks -->
    <div class="card green-accent">
      <div class="card-title"><span class="icon">&#127975;</span> Central Bank Reserves</div>
      <div style="display:flex;gap:16px;margin:12px 0">
        <div style="flex:1;padding:12px;background:var(--bg-elevated);border-radius:8px;text-align:center">
          <div style="font-size:10px;color:var(--text-muted)">PBOC Streak</div>
          <div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono';color:var(--green)">{cb['pboc']['consecutive_months']}</div>
          <div style="font-size:10px;color:var(--text-muted)">months</div>
        </div>
        <div style="flex:1;padding:12px;background:var(--bg-elevated);border-radius:8px;text-align:center">
          <div style="font-size:10px;color:var(--text-muted)">PBOC Holdings</div>
          <div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono';color:var(--gold)">{cb['pboc']['total_tonnes']:,}</div>
          <div style="font-size:10px;color:var(--text-muted)">tonnes</div>
        </div>
        <div style="flex:1;padding:12px;background:var(--bg-elevated);border-radius:8px;text-align:center">
          <div style="font-size:10px;color:var(--text-muted)">RBI Holdings</div>
          <div style="font-size:28px;font-weight:700;font-family:'JetBrains Mono';color:var(--orange)">{cb['rbi']['total_tonnes']:,}</div>
          <div style="font-size:10px;color:var(--text-muted)">tonnes</div>
        </div>
      </div>
      <div class="metric-row">
        <span class="metric-label">PBOC Reserve Ratio</span>
        <span class="metric-value">{cb['pboc']['reserve_pct']}% <span style="color:var(--text-muted);font-size:10px">(G7 avg: 60-70%)</span></span>
      </div>
      <div class="metric-row">
        <span class="metric-label">PBOC US Treasury</span>
        <span class="metric-value negative">${cb['pboc']['usd_holdings_b']}B (-11% YoY)</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Global CB Demand</span>
        <span class="metric-value positive">~{cb['annual_demand_tonnes']}t/yr (25-30% of mine supply)</span>
      </div>
      <div style="margin-top:8px;padding:6px;background:var(--green-dim);border-radius:4px;font-size:10px;color:var(--green)">
        Structural floor: CB buying is price-insensitive = downside protection
      </div>
    </div>

    <!-- China Speculative -->
    <div class="card red-accent" style="border-color:rgba(255,145,0,0.2)">
      <div class="card-title"><span class="icon">&#127464;&#127475;</span> China Speculative Flows <span class="tag warn">RISK</span></div>
      <div style="display:flex;gap:16px;margin:12px 0">
        <div style="flex:1;padding:12px;background:var(--bg-elevated);border-radius:8px;text-align:center">
          <div style="font-size:10px;color:var(--text-muted)">SHFE Vol/Day</div>
          <div style="font-size:22px;font-weight:700;font-family:'JetBrains Mono';color:var(--orange)">{ch['shfe_volume_tonnes_day']}</div>
          <div style="font-size:10px;color:var(--text-muted)">tonnes</div>
        </div>
        <div style="flex:1;padding:12px;background:var(--bg-elevated);border-radius:8px;text-align:center">
          <div style="font-size:10px;color:var(--text-muted)">ETF Jan Flow</div>
          <div style="font-size:22px;font-weight:700;font-family:'JetBrains Mono';color:var(--green)">${ch['etf_jan_flow_usd_b']}B</div>
          <div style="font-size:10px;color:var(--text-muted)">record YTD</div>
        </div>
        <div style="flex:1;padding:12px;background:var(--bg-elevated);border-radius:8px;text-align:center">
          <div style="font-size:10px;color:var(--text-muted)">SGE Withdrawal</div>
          <div style="font-size:22px;font-weight:700;font-family:'JetBrains Mono';color:var(--gold)">{ch['sge_withdrawal_tonnes']}</div>
          <div style="font-size:10px;color:var(--text-muted)">tonnes/Jan</div>
        </div>
      </div>
      <div class="metric-row">
        <span class="metric-label">Household Gold Alloc</span>
        <span class="metric-value">{ch['household_gold_pct']}% <span style="color:var(--text-muted);font-size:10px">target: 5%</span></span>
      </div>
      <div style="margin-top:8px;padding:6px;background:rgba(255,145,0,0.1);border-radius:4px;font-size:10px;color:var(--orange)">
        Capital Economics warns of speculative bubble risk<br>
        SGE has raised margins repeatedly
      </div>
    </div>

  </div>

  <!-- Row 5: Silver Supply + Trading Plans -->
  <div class="section-title">Silver Supply Deficit</div>
  <div class="grid grid-4">
    <div class="card silver-accent" style="grid-column: span 2">
      <div class="card-title"><span class="icon">&#9874;</span> Silver Structural Deficit (6th Year)</div>
      <div style="display:flex;align-items:center;gap:20px;margin:8px 0">
        <div style="text-align:center">
          <div style="font-size:32px;font-weight:700;font-family:'JetBrains Mono';color:var(--red)">{ss['deficit_2026_moz']}</div>
          <div style="font-size:10px;color:var(--text-muted)">2026 Deficit (Moz)</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:32px;font-weight:700;font-family:'JetBrains Mono';color:var(--red)">{ss['cumulative_deficit_moz']}</div>
          <div style="font-size:10px;color:var(--text-muted)">Cumulative (Moz)</div>
        </div>
      </div>
      <div class="bar-chart">
        <div class="bar-row"><span class="bar-label">2021</span><div class="bar-track"><div class="bar-fill negative" style="width:28%">-51</div></div></div>
        <div class="bar-row"><span class="bar-label">2022</span><div class="bar-track"><div class="bar-fill negative" style="width:100%">-180</div></div></div>
        <div class="bar-row"><span class="bar-label">2023</span><div class="bar-track"><div class="bar-fill negative" style="width:83%">-150</div></div></div>
        <div class="bar-row"><span class="bar-label">2024</span><div class="bar-track"><div class="bar-fill negative" style="width:67%">-120</div></div></div>
        <div class="bar-row"><span class="bar-label">2025</span><div class="bar-track"><div class="bar-fill negative" style="width:53%">-95</div></div></div>
        <div class="bar-row"><span class="bar-label">2026E</span><div class="bar-track"><div class="bar-fill negative" style="width:37%">-67</div></div></div>
      </div>
      <div style="margin-top:8px;font-size:10px;color:var(--text-muted)">
        Cumulative ~870 Moz deficit = ~1 year of mine production<br>
        China export controls (Jan 2026): only 44 firms licensed, 60-70% of refined supply
      </div>
    </div>

    <div class="card" style="grid-column: span 2;background: linear-gradient(135deg, var(--bg-card) 0%, rgba(18,18,26,0.8) 100%)">
      <div class="card-title"><span class="icon">&#9888;</span> Key Risk Monitors</div>
      <div class="metric-row">
        <span class="metric-label">TIPS &gt; 2.00%</span>
        <span class="metric-value">Reduce gold longs</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">DXY &gt; 100</span>
        <span class="metric-value">Hedge / reduce</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">CME Margin Hike</span>
        <span class="metric-value negative">Forced liq risk (Silver!)</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">China Regulation</span>
        <span class="metric-value negative">Demand evaporation</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">COT Extreme Long</span>
        <span class="metric-value">Sentiment top risk</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Powell Successor</span>
        <span class="metric-value">Hawkish = bearish</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Max Portfolio Risk</span>
        <span class="metric-value purple-color">5% of capital (precious metals)</span>
      </div>
    </div>
  </div>

  <!-- Row 6: Trading Plans -->
  <div class="section-title">Trading Plans</div>
  <div class="grid grid-2">

    <!-- Gold Plan -->
    <div class="card gold-accent">
      <div class="card-title"><span class="icon">Au</span> Gold Trading Plan</div>

      <div style="margin-bottom:16px">
        <div style="display:flex;gap:6px;margin-bottom:10px">
          <span class="tag long">LONG</span>
          <span class="tag main">MAIN</span>
        </div>
        <div class="trade-entry">
          <span class="trade-label">Entry</span><span class="trade-val entry">{tp['gold_long']['entry']}</span>
          <span class="trade-label">Alt</span><span class="trade-val entry">{tp['gold_long']['alt_entry']}</span>
          <span class="trade-label">TP1</span><span class="trade-val tp">${tp['gold_long']['tp1']:,.0f}</span>
          <span class="trade-label">TP2</span><span class="trade-val tp">${tp['gold_long']['tp2']:,.0f}</span>
          <span class="trade-label">TP3</span><span class="trade-val tp">${tp['gold_long']['tp3']:,.0f} (ATH)</span>
          <span class="trade-label">SL</span><span class="trade-val sl">${tp['gold_long']['sl']:,.0f}</span>
          <span class="trade-label">Size</span><span class="trade-val">2-3% risk</span>
          <span class="trade-label">R:R</span><span class="trade-val">~1:2.5</span>
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:12px">
        <div style="display:flex;gap:6px;margin-bottom:10px">
          <span class="tag short">SHORT</span>
          <span class="tag sub">SUB</span>
        </div>
        <div class="trade-entry">
          <span class="trade-label">Entry</span><span class="trade-val entry">{tp['gold_short']['entry']}</span>
          <span class="trade-label">TP1</span><span class="trade-val tp">${tp['gold_short']['tp1']:,.0f}</span>
          <span class="trade-label">TP2</span><span class="trade-val tp">${tp['gold_short']['tp2']:,.0f}</span>
          <span class="trade-label">SL</span><span class="trade-val sl">${tp['gold_short']['sl']:,.0f}</span>
          <span class="trade-label">Size</span><span class="trade-val">1-1.5% risk</span>
        </div>
      </div>
    </div>

    <!-- Silver Plan -->
    <div class="card silver-accent">
      <div class="card-title"><span class="icon">Ag</span> Silver Trading Plan</div>

      <div style="margin-bottom:16px">
        <div style="display:flex;gap:6px;margin-bottom:10px">
          <span class="tag long">LONG</span>
          <span class="tag main">MAIN</span>
        </div>
        <div class="trade-entry">
          <span class="trade-label">Entry1</span><span class="trade-val entry">{tp['silver_long']['entry1']}</span>
          <span class="trade-label">Entry2</span><span class="trade-val entry">{tp['silver_long']['entry2']}</span>
          <span class="trade-label">TP1</span><span class="trade-val tp">${tp['silver_long']['tp1']:,.2f}</span>
          <span class="trade-label">TP2</span><span class="trade-val tp">${tp['silver_long']['tp2']:,.2f}</span>
          <span class="trade-label">TP3</span><span class="trade-val tp">${tp['silver_long']['tp3']:,.2f}</span>
          <span class="trade-label">SL</span><span class="trade-val sl">${tp['silver_long']['sl']:,.2f}</span>
          <span class="trade-label">Size</span><span class="trade-val">1.5-2% risk</span>
          <span class="trade-label">R:R</span><span class="trade-val">~1:2.2 / 1:2.8</span>
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:12px">
        <div style="display:flex;gap:6px;margin-bottom:10px">
          <span class="tag short">SHORT</span>
          <span class="tag sub">SUB</span>
        </div>
        <div class="trade-entry">
          <span class="trade-label">Entry</span><span class="trade-val entry">{tp['silver_short']['entry']}</span>
          <span class="trade-label">TP1</span><span class="trade-val tp">${tp['silver_short']['tp1']:,.2f}</span>
          <span class="trade-label">TP2</span><span class="trade-val tp">${tp['silver_short']['tp2']:,.2f}</span>
          <span class="trade-label">SL</span><span class="trade-val sl">${tp['silver_short']['sl']:,.2f}</span>
          <span class="trade-label">Size</span><span class="trade-val">1% risk</span>
        </div>
        <div style="margin-top:8px;padding:6px;background:rgba(255,145,0,0.1);border-radius:4px;font-size:10px;color:var(--orange)">
          CME Silver margin 18% (2x Gold). Consider SLV/options instead of futures.
        </div>
      </div>
    </div>

  </div>

  <!-- Footer -->
  <div class="footer">
    Disclaimer: This dashboard is for informational purposes only, not investment advice.<br>
    Generated by Gold &amp; Silver Dashboard v1.0 | Data sources: FRED, CFTC, WGC, CME, ETFdb<br>
    Last updated: {d['generated_date']} {d['generated_time']}
  </div>

</div>
</body>
</html>"""

    return html


def main():
    data = load_data()
    html = generate_html(data)

    out_path = os.path.join(OUTPUT_DIR, "gold_silver_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard generated: {out_path}")

    # Also archive
    date_str = data.get("generated_date", datetime.date.today().isoformat())
    archive_path = os.path.join(OUTPUT_DIR, "archive", f"dashboard_{date_str}.html")
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Archive saved:       {archive_path}")


if __name__ == "__main__":
    main()

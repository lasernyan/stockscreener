"""
XAUUSD Regime-Adaptive Swing Strategy - Python Backtest
========================================================
Pine Script 戦略の完全な Python 再実装 + バックテスト

依存: pandas, numpy, yfinance, matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS (Pine Script inputs と同一)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyParams:
    # Capital & Costs
    initial_capital: float = 100_000
    position_pct: float = 0.03        # 3% of equity per trade
    commission_pct: float = 0.0002     # 0.02%
    slippage_pts: float = 2.0

    # Macro Trend Filter
    use_macro_filter: bool = True
    macro_ema_len: int = 50            # Weekly EMA period

    # Regime Detection
    er_len: int = 10
    er_smooth: int = 5
    er_threshold: float = 0.40

    # Trend Mode (Pullback)
    use_trend_mode: bool = True
    trend_ema_len: int = 21
    pullback_atr_mult: float = 1.2
    trend_rsi_len: int = 14
    trend_rsi_ob: float = 55.0        # Long RSI upper limit
    trend_rsi_os: float = 45.0        # Short RSI lower limit

    # Range Mode (Mean Reversion)
    use_range_mode: bool = True
    bb_len: int = 20
    bb_mult: float = 2.0
    range_rsi_len: int = 14
    range_rsi_ob: float = 70.0
    range_rsi_os: float = 30.0

    # Risk Management
    atr_len: int = 14
    init_sl_atr: float = 2.0
    use_chandelier: bool = True
    chandelier_mult: float = 3.0
    use_partial_tp: bool = True
    partial_tp_atr: float = 2.5
    use_deviation_tp: bool = True
    ma200_len: int = 200
    tp_deviation_pct: float = 10.0

    # Volatility Adaptation
    use_vol_adapt: bool = True
    atr_pctile_len: int = 100
    high_vol_pctile: float = 90.0
    high_vol_tighten: float = 0.7

    # Time Stop
    use_time_stop: bool = True
    max_bars_in_trade: int = 30
    min_profit_atr: float = 0.5

    # Direction
    long_only: bool = False
    short_only: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int, mult: float):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid, mid + mult * std, mid - mult * std


def kaufman_er(close: pd.Series, length: int, smooth: int) -> pd.Series:
    direction = (close - close.shift(length)).abs()
    volatility = close.diff().abs().rolling(length).sum()
    raw_er = direction / volatility.replace(0, np.nan)
    raw_er = raw_er.fillna(0)
    return ema(raw_er, smooth)


def percentrank(series: pd.Series, length: int) -> pd.Series:
    def _rank(window):
        if len(window) < 2:
            return 50.0
        current = window.iloc[-1]
        count_below = (window.iloc[:-1] < current).sum()
        return count_below / (len(window) - 1) * 100
    return series.rolling(length).apply(_rank, raw=False)


def resample_weekly_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """日足→週足にリサンプルしてEMAを計算し、日足に再マップ"""
    weekly = df["Close"].resample("W-FRI").last().dropna()
    weekly_ema = ema(weekly, period)
    # 週足EMAを日足に前方フィル（look-ahead なし）
    return weekly_ema.reindex(df.index, method="ffill")


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════

def fetch_data(symbol: str = "GC=F", start: str = "2005-01-01",
               end: str = "2026-02-10") -> pd.DataFrame:
    """
    yfinance で XAUUSD (Gold Futures) の日足データを取得。
    取得失敗時はゴールドの統計的性質に基づくシミュレーションデータを生成。
    """
    try:
        print(f"Fetching {symbol} data from {start} to {end} ...")
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) > 100:
            print(f"  -> {len(df)} bars loaded ({df.index[0].date()} ~ {df.index[-1].date()})")
            return df
    except Exception as e:
        print(f"  yfinance failed: {e}")

    print("  -> Generating synthetic XAUUSD data (realistic GBM + regime model) ...")
    return generate_synthetic_xauusd(start, end)


def generate_synthetic_xauusd(start: str, end: str) -> pd.DataFrame:
    """
    XAUUSD の歴史的パターンに基づくリアルな合成日足データ生成。
    実際のゴールド相場に近い「上昇→暴落→横ばい→再上昇」のマクロサイクルと
    トレンド/レンジのレジーム切替を再現。
    """
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # ── マクロサイクル: 歴史的XAUUSDパスを近似 ────────────────────────────
    # 2005: ~$420 → 2008: ~$900 (上昇) → 2008 crash: ~$700
    # 2009-2011: ~$700 → $1920 (大上昇) → 2011-2015: $1920→$1050 (大下落)
    # 2016-2019: $1050→$1550 (緩やか上昇) → 2020-2024: $1550→$2700 (上昇)
    macro_phases = []
    # (start_frac, end_frac, annual_drift, annual_vol)
    phases = [
        (0.00, 0.14, 0.25, 0.18),   # 2005-2008: 上昇
        (0.14, 0.16, -0.50, 0.40),   # 2008 crash
        (0.16, 0.32, 0.35, 0.20),    # 2009-2011: 大上昇
        (0.32, 0.50, -0.15, 0.22),   # 2012-2015: 下落
        (0.50, 0.65, 0.12, 0.14),    # 2016-2019: 緩やか上昇
        (0.65, 0.78, 0.22, 0.18),    # 2020-2022: 上昇
        (0.78, 0.85, -0.05, 0.16),   # 2022-2023: 横ばい
        (0.85, 1.00, 0.30, 0.16),    # 2024-2026: 再上昇
    ]

    annual_drift_arr = np.zeros(n)
    annual_vol_arr = np.zeros(n)
    for start_f, end_f, drift, vol in phases:
        i_start = int(start_f * n)
        i_end = int(end_f * n)
        annual_drift_arr[i_start:i_end] = drift
        annual_vol_arr[i_start:i_end] = vol

    # Markov regime overlay (短期のトレンド/レンジ切替)
    regime = np.zeros(n, dtype=int)  # 0=trend, 1=range
    for i in range(1, n):
        if regime[i-1] == 0:
            regime[i] = 0 if np.random.random() < 0.96 else 1  # avg 25 days trend
        else:
            regime[i] = 1 if np.random.random() < 0.93 else 0  # avg 14 days range

    # レジームによるボラティリティ微調整
    vol_mult = np.where(regime == 0, 0.85, 1.3)

    daily_drift = annual_drift_arr / 252
    daily_vol = annual_vol_arr / np.sqrt(252) * vol_mult

    # 日次リターン生成 (fat tails: t-distribution)
    raw_noise = np.random.standard_t(df=5, size=n) / np.sqrt(5/3)  # 正規化
    log_returns = daily_drift + daily_vol * raw_noise

    # 短期自己相関 (momentum effect)
    for i in range(1, n):
        log_returns[i] += 0.05 * log_returns[i-1]

    # 価格パス生成
    close_prices = np.zeros(n)
    close_prices[0] = 420.0
    for i in range(1, n):
        close_prices[i] = close_prices[i-1] * np.exp(log_returns[i])

    # OHLC生成 (日中レンジをリアルに)
    intraday_vol = daily_vol * close_prices
    high_offsets = np.abs(np.random.randn(n)) * intraday_vol * 0.7
    low_offsets  = np.abs(np.random.randn(n)) * intraday_vol * 0.7
    open_offsets = np.random.randn(n) * intraday_vol * 0.3

    open_prices  = close_prices + open_offsets
    high_prices  = np.maximum(open_prices, close_prices) + high_offsets
    low_prices   = np.minimum(open_prices, close_prices) - low_offsets

    # Volume
    base_volume = 150000
    volume = (base_volume * (1 + np.abs(log_returns) / 0.01) *
              np.random.uniform(0.6, 1.4, n)).astype(int)

    df = pd.DataFrame({
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume
    }, index=dates)

    print(f"  -> {len(df)} synthetic bars generated ({df.index[0].date()} ~ {df.index[-1].date()})")
    print(f"     Price range: ${df['Close'].min():.0f} ~ ${df['Close'].max():.0f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date: pd.Timestamp
    direction: int              # +1 = long, -1 = short
    entry_price: float
    qty: float
    entry_atr: float
    mode: str                   # "Trend" or "Range"
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    partial_taken: bool = False


def run_backtest(df: pd.DataFrame, p: StrategyParams) -> dict:
    """
    Pine Script のロジックをバーごとにシミュレーション
    """
    # ── Indicator pre-computation ─────────────────────────────────────────
    df = df.copy()
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], p.atr_len)
    df["MA200"] = sma(df["Close"], p.ma200_len)
    df["DeviationPct"] = (df["Close"] - df["MA200"]) / df["MA200"] * 100
    df["SmoothER"] = kaufman_er(df["Close"], p.er_len, p.er_smooth)
    df["IsTrending"] = df["SmoothER"] >= p.er_threshold
    df["TrendEMA"] = ema(df["Close"], p.trend_ema_len)
    df["TrendRSI"] = rsi(df["Close"], p.trend_rsi_len)
    df["RangeRSI"] = rsi(df["Close"], p.range_rsi_len)
    bb_mid, bb_upper, bb_lower = bollinger_bands(df["Close"], p.bb_len, p.bb_mult)
    df["BB_Upper"] = bb_upper
    df["BB_Lower"] = bb_lower
    df["ATR_Rank"] = percentrank(df["ATR"], p.atr_pctile_len)
    df["IsHighVol"] = df["ATR_Rank"] >= p.high_vol_pctile

    # Weekly EMA (macro filter)
    df["WeeklyEMA"] = resample_weekly_ema(df, p.macro_ema_len)

    # EMA slope (3-bar lookback)
    df["EmaSlopeUp"] = df["TrendEMA"] > df["TrendEMA"].shift(3)
    df["EmaSlopeDown"] = df["TrendEMA"] < df["TrendEMA"].shift(3)

    # Pullback zone
    df["DistFromEma"] = (df["Close"] - df["TrendEMA"]).abs()
    df["InPullbackZone"] = df["DistFromEma"] <= p.pullback_atr_mult * df["ATR"]

    # Pre-compute highest/lowest rolling for chandelier
    df["RollingHigh"] = df["High"].rolling(p.atr_len, min_periods=1).max()
    df["RollingLow"] = df["Low"].rolling(p.atr_len, min_periods=1).min()

    # ── Simulation ────────────────────────────────────────────────────────
    equity = p.initial_capital
    equity_curve = []
    trades: list[Trade] = []
    current_trade: Optional[Trade] = None
    bars_in_trade = 0
    trail_stop = np.nan
    initial_sl = np.nan

    warmup = max(p.ma200_len, p.atr_pctile_len, p.macro_ema_len * 5) + 10

    # Diagnostic counters
    diag = {"trending_bars": 0, "ranging_bars": 0,
            "trend_long_raw": 0, "trend_short_raw": 0,
            "range_long_raw": 0, "range_short_raw": 0,
            "macro_filtered_long": 0, "macro_filtered_short": 0,
            "entries_long": 0, "entries_short": 0}

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        date = df.index[i]
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        atr_val = row["ATR"]

        if i < warmup or pd.isna(atr_val) or atr_val <= 0:
            equity_curve.append(equity)
            continue

        # ── Position Management (check stops first) ──────────────────────
        if current_trade is not None:
            bars_in_trade += 1
            d = current_trade.direction

            # Adaptive chandelier multiplier
            if p.use_vol_adapt and row["IsHighVol"]:
                adapt_mult = p.chandelier_mult * p.high_vol_tighten
            else:
                adapt_mult = p.chandelier_mult

            # Chandelier trailing stop update
            if p.use_chandelier:
                if d == 1:
                    ch_level = row["RollingHigh"] - adapt_mult * atr_val
                    if not np.isnan(trail_stop):
                        trail_stop = max(trail_stop, ch_level)
                    else:
                        trail_stop = ch_level
                else:
                    ch_level = row["RollingLow"] + adapt_mult * atr_val
                    if not np.isnan(trail_stop):
                        trail_stop = min(trail_stop, ch_level)
                    else:
                        trail_stop = ch_level

            # Clamp to initial SL
            if d == 1:
                trail_stop = max(trail_stop, initial_sl)
            else:
                trail_stop = min(trail_stop, initial_sl)

            # Check stop hit (intra-bar)
            stopped = False
            if d == 1 and l <= trail_stop:
                exit_px = trail_stop + p.slippage_pts
                stopped = True
                reason = "Stop"
            elif d == -1 and h >= trail_stop:
                exit_px = trail_stop - p.slippage_pts
                stopped = True
                reason = "Stop"

            # Partial TP check
            if not stopped and p.use_partial_tp and not current_trade.partial_taken:
                tp_level_hit = False
                if d == 1:
                    dev_ok = (not p.use_deviation_tp) or (row["DeviationPct"] >= p.tp_deviation_pct)
                    if h >= current_trade.entry_price + p.partial_tp_atr * current_trade.entry_atr and dev_ok:
                        tp_level_hit = True
                else:
                    dev_ok = (not p.use_deviation_tp) or (row["DeviationPct"] <= -p.tp_deviation_pct)
                    if l <= current_trade.entry_price - p.partial_tp_atr * current_trade.entry_atr and dev_ok:
                        tp_level_hit = True

                if tp_level_hit:
                    # Realize 50% at TP level
                    tp_price = current_trade.entry_price + d * p.partial_tp_atr * current_trade.entry_atr
                    half_qty = current_trade.qty * 0.5
                    pnl_partial = d * (tp_price - current_trade.entry_price) * half_qty
                    commission = tp_price * half_qty * p.commission_pct * 2
                    equity += pnl_partial - commission
                    current_trade.qty -= half_qty
                    current_trade.partial_taken = True
                    # Move stop to entry
                    if d == 1:
                        trail_stop = max(trail_stop, current_trade.entry_price)
                    else:
                        trail_stop = min(trail_stop, current_trade.entry_price)

            # Time stop
            if not stopped and p.use_time_stop and bars_in_trade >= p.max_bars_in_trade:
                profit = d * (c - current_trade.entry_price)
                if profit < p.min_profit_atr * current_trade.entry_atr:
                    exit_px = c + p.slippage_pts * (-d)
                    stopped = True
                    reason = "TimeStop"

            # Execute stop/exit
            if stopped:
                remaining_pnl = d * (exit_px - current_trade.entry_price) * current_trade.qty
                commission = exit_px * current_trade.qty * p.commission_pct * 2
                equity += remaining_pnl - commission
                current_trade.exit_date = date
                current_trade.exit_price = exit_px
                current_trade.exit_reason = reason
                current_trade.pnl = remaining_pnl - commission
                if current_trade.partial_taken:
                    # Add back partial PnL for total
                    pass  # Already added to equity above
                trades.append(current_trade)
                current_trade = None
                bars_in_trade = 0

        # ── Signal Generation ─────────────────────────────────────────────
        if current_trade is None and i >= warmup:
            is_trending = row["IsTrending"]
            is_ranging = not is_trending

            # Macro filter
            weekly_ema_val = row["WeeklyEMA"]
            macro_long_ok = (not p.use_macro_filter) or (pd.isna(weekly_ema_val) or c > weekly_ema_val)
            macro_short_ok = (not p.use_macro_filter) or (pd.isna(weekly_ema_val) or c < weekly_ema_val)

            # Trend mode signals
            # Pine: close > trendEma and close[1] <= trendEma
            # close[1] は前足close, trendEma は現足EMA (EMAクロス)
            trend_long = (p.use_trend_mode and is_trending and
                         row["EmaSlopeUp"] and row["InPullbackZone"] and
                         row["TrendRSI"] < p.trend_rsi_ob and
                         c > row["TrendEMA"] and prev["Close"] <= row["TrendEMA"])

            trend_short = (p.use_trend_mode and is_trending and
                          row["EmaSlopeDown"] and row["InPullbackZone"] and
                          row["TrendRSI"] > p.trend_rsi_os and
                          c < row["TrendEMA"] and prev["Close"] >= row["TrendEMA"])

            # Range mode signals
            range_long = (p.use_range_mode and is_ranging and
                         prev["Close"] <= prev["BB_Lower"] and
                         prev["RangeRSI"] < p.range_rsi_os and
                         c > o)  # bullish confirmation candle

            range_short = (p.use_range_mode and is_ranging and
                          prev["Close"] >= prev["BB_Upper"] and
                          prev["RangeRSI"] > p.range_rsi_ob and
                          c < o)  # bearish confirmation candle

            # Diagnostics
            if is_trending:
                diag["trending_bars"] += 1
            else:
                diag["ranging_bars"] += 1
            if trend_long: diag["trend_long_raw"] += 1
            if trend_short: diag["trend_short_raw"] += 1
            if range_long: diag["range_long_raw"] += 1
            if range_short: diag["range_short_raw"] += 1

            # Combined
            long_signal = (trend_long or range_long) and macro_long_ok and not p.short_only
            short_signal = (trend_short or range_short) and macro_short_ok and not p.long_only

            if (trend_long or range_long) and not macro_long_ok:
                diag["macro_filtered_long"] += 1
            if (trend_short or range_short) and not macro_short_ok:
                diag["macro_filtered_short"] += 1

            # Entry
            if long_signal:
                diag["entries_long"] += 1
                entry_px = c + p.slippage_pts
                qty = (equity * p.position_pct) / entry_px
                mode = "Trend" if trend_long else "Range"
                current_trade = Trade(
                    entry_date=date, direction=1, entry_price=entry_px,
                    qty=qty, entry_atr=atr_val, mode=mode
                )
                initial_sl = entry_px - p.init_sl_atr * atr_val
                trail_stop = initial_sl
                bars_in_trade = 0
                commission = entry_px * qty * p.commission_pct
                equity -= commission

            elif short_signal:
                diag["entries_short"] += 1
                entry_px = c - p.slippage_pts
                qty = (equity * p.position_pct) / entry_px
                mode = "Trend" if trend_short else "Range"
                current_trade = Trade(
                    entry_date=date, direction=-1, entry_price=entry_px,
                    qty=qty, entry_atr=atr_val, mode=mode
                )
                initial_sl = entry_px + p.init_sl_atr * atr_val
                trail_stop = initial_sl
                bars_in_trade = 0
                commission = entry_px * qty * p.commission_pct
                equity -= commission

        # Mark-to-market equity
        if current_trade is not None:
            d = current_trade.direction
            unrealized = d * (c - current_trade.entry_price) * current_trade.qty
            equity_curve.append(equity + unrealized)
        else:
            equity_curve.append(equity)

    # Close any remaining position at last bar
    if current_trade is not None:
        last = df.iloc[-1]
        c = last["Close"]
        d = current_trade.direction
        remaining_pnl = d * (c - current_trade.entry_price) * current_trade.qty
        commission = c * current_trade.qty * p.commission_pct * 2
        equity += remaining_pnl - commission
        current_trade.exit_date = df.index[-1]
        current_trade.exit_price = c
        current_trade.exit_reason = "EndOfData"
        current_trade.pnl = remaining_pnl - commission
        trades.append(current_trade)

    # Pad equity curve to match DataFrame length
    while len(equity_curve) < len(df):
        equity_curve.insert(0, p.initial_capital)

    df["Equity"] = equity_curve

    return {"df": df, "trades": trades, "final_equity": equity_curve[-1],
            "params": p, "diag": diag}


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def analyze(result: dict):
    trades = result["trades"]
    p = result["params"]
    df = result["df"]
    eq = df["Equity"]

    if not trades:
        print("No trades generated.")
        return

    pnls = [t.pnl for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    # Drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()

    # CAGR
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((result["final_equity"] / p.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Holding period
    durations = []
    for t in trades:
        if t.exit_date and t.entry_date:
            durations.append((t.exit_date - t.entry_date).days)
    avg_hold = np.mean(durations) if durations else 0

    # Sharpe (annualized, daily returns)
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Trade mode breakdown
    trend_trades = [t for t in trades if t.mode == "Trend"]
    range_trades = [t for t in trades if t.mode == "Range"]
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    # Exit reason breakdown
    from collections import Counter
    exit_reasons = Counter(t.exit_reason for t in trades)

    print("\n" + "=" * 65)
    print("  XAUUSD Regime-Adaptive Swing Strategy - Backtest Results")
    print("=" * 65)
    print(f"  Period        : {df.index[0].date()} ~ {df.index[-1].date()} ({years:.1f} years)")
    print(f"  Initial Capital: ${p.initial_capital:,.0f}")
    print(f"  Final Equity   : ${result['final_equity']:,.0f}")
    print(f"  Total Return   : {(result['final_equity']/p.initial_capital - 1)*100:+.2f}%")
    print(f"  CAGR           : {cagr:+.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Max Drawdown   : {max_dd:.2f}%")
    print("-" * 65)
    print(f"  Total Trades   : {len(trades)}")
    print(f"  Win Rate       : {win_rate:.1f}%")
    print(f"  Avg Win        : ${avg_win:,.2f}")
    print(f"  Avg Loss       : ${avg_loss:,.2f}")
    print(f"  Profit Factor  : {profit_factor:.2f}")
    print(f"  Avg Hold (days): {avg_hold:.1f}")
    print("-" * 65)
    print(f"  Trend Trades   : {len(trend_trades)}  "
          f"(Win {sum(1 for t in trend_trades if t.pnl>0)}/{len(trend_trades)})")
    print(f"  Range Trades   : {len(range_trades)}  "
          f"(Win {sum(1 for t in range_trades if t.pnl>0)}/{len(range_trades)})")
    print(f"  Long Trades    : {len(long_trades)}  "
          f"(Win {sum(1 for t in long_trades if t.pnl>0)}/{len(long_trades)})")
    print(f"  Short Trades   : {len(short_trades)}  "
          f"(Win {sum(1 for t in short_trades if t.pnl>0)}/{len(short_trades)})")
    print("-" * 65)
    print("  Exit Reasons:")
    for reason, count in exit_reasons.most_common():
        print(f"    {reason:12s}: {count}")
    print("=" * 65)

    # Diagnostics
    diag = result.get("diag", {})
    if diag:
        print("\n  Signal Diagnostics:")
        print(f"    Trending bars   : {diag.get('trending_bars', 0)}")
        print(f"    Ranging bars    : {diag.get('ranging_bars', 0)}")
        print(f"    Trend Long raw  : {diag.get('trend_long_raw', 0)}")
        print(f"    Trend Short raw : {diag.get('trend_short_raw', 0)}")
        print(f"    Range Long raw  : {diag.get('range_long_raw', 0)}")
        print(f"    Range Short raw : {diag.get('range_short_raw', 0)}")
        print(f"    Macro filtered L: {diag.get('macro_filtered_long', 0)}")
        print(f"    Macro filtered S: {diag.get('macro_filtered_short', 0)}")
        print(f"    Entries Long    : {diag.get('entries_long', 0)}")
        print(f"    Entries Short   : {diag.get('entries_short', 0)}")
        print("=" * 65)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[2, 1, 1],
                             gridspec_kw={"hspace": 0.3})

    # 1) Equity curve
    ax1 = axes[0]
    ax1.plot(df.index, eq, color="royalblue", linewidth=1.2, label="Equity")
    ax1.fill_between(df.index, p.initial_capital, eq,
                     where=eq >= p.initial_capital, alpha=0.15, color="green")
    ax1.fill_between(df.index, p.initial_capital, eq,
                     where=eq < p.initial_capital, alpha=0.15, color="red")
    ax1.axhline(p.initial_capital, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("Equity Curve", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2) Drawdown
    ax2 = axes[1]
    ax2.fill_between(df.index, 0, dd, color="crimson", alpha=0.4)
    ax2.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("DD %")
    ax2.grid(True, alpha=0.3)

    # 3) Price + entry/exit markers
    ax3 = axes[2]
    ax3.plot(df.index, df["Close"], color="gold", linewidth=0.8, label="XAUUSD Close")
    ax3.plot(df.index, df["TrendEMA"], color="cyan", linewidth=0.6, alpha=0.7, label=f"EMA{p.trend_ema_len}")

    for t in trades:
        color = "lime" if t.direction == 1 else "red"
        marker = "^" if t.direction == 1 else "v"
        ax3.scatter(t.entry_date, t.entry_price, color=color, marker=marker,
                   s=30, zorder=5, alpha=0.8)
        if t.exit_date:
            ec = "blue" if t.pnl > 0 else "orange"
            ax3.scatter(t.exit_date, t.exit_price, color=ec, marker="x",
                       s=25, zorder=5, alpha=0.8)

    ax3.set_title("XAUUSD Price & Trade Entries", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Price ($)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved to: backtest_result.png")
    plt.close()

    # ── Trade list (last 20) ─────────────────────────────────────────────
    print(f"\n  Last 20 Trades:")
    print(f"  {'Date':12s} {'Dir':5s} {'Mode':6s} {'Entry':>9s} {'Exit':>9s} "
          f"{'PnL':>10s} {'Hold':>5s} {'Reason':>10s}")
    print("  " + "-" * 70)
    for t in trades[-20:]:
        d = "LONG" if t.direction == 1 else "SHORT"
        hold = (t.exit_date - t.entry_date).days if t.exit_date else 0
        print(f"  {t.entry_date.strftime('%Y-%m-%d'):12s} {d:5s} {t.mode:6s} "
              f"{t.entry_price:9.2f} {t.exit_price:9.2f} "
              f"{'$'+f'{t.pnl:,.2f}':>10s} {hold:5d} {t.exit_reason:>10s}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario_comparison(df: pd.DataFrame):
    """複数のパラメータ設定でバックテストを実行し、比較テーブルを表示"""
    scenarios = {
        "A: Full (default)": StrategyParams(),
        "B: No Macro Filter": StrategyParams(use_macro_filter=False),
        "C: No Macro + No VolAdapt": StrategyParams(use_macro_filter=False, use_vol_adapt=False),
        "D: Trend Only": StrategyParams(use_macro_filter=False, use_range_mode=False),
        "E: Range Only": StrategyParams(use_macro_filter=False, use_trend_mode=False),
        "F: No Partial TP": StrategyParams(use_macro_filter=False, use_partial_tp=False),
        "G: Long Only (No Macro)": StrategyParams(use_macro_filter=False, long_only=True),
    }

    rows = []
    best_result = None
    best_name = ""

    for name, params in scenarios.items():
        r = run_backtest(df, params)
        trades = r["trades"]
        eq = r["df"]["Equity"]
        pnls = [t.pnl for t in trades]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        peak = eq.expanding().max()
        dd = ((eq - peak) / peak * 100).min()
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = ((r["final_equity"] / params.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        daily_ret = eq.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
        durations = [(t.exit_date - t.entry_date).days for t in trades if t.exit_date and t.entry_date]
        avg_hold = np.mean(durations) if durations else 0

        rows.append({
            "name": name, "trades": len(trades),
            "win_rate": len(wins)/len(trades)*100 if trades else 0,
            "pf": pf, "cagr": cagr, "sharpe": sharpe,
            "max_dd": dd, "return_pct": (r["final_equity"]/params.initial_capital - 1)*100,
            "avg_hold": avg_hold
        })

        if len(trades) > 0 and (best_result is None or r["final_equity"] > best_result["final_equity"]):
            best_result = r
            best_name = name

    print("\n" + "=" * 100)
    print("  SCENARIO COMPARISON")
    print("=" * 100)
    print(f"  {'Scenario':<28s} {'Trades':>6s} {'WinRate':>8s} {'PF':>6s} "
          f"{'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Return':>8s} {'AvgHold':>8s}")
    print("  " + "-" * 92)
    for r in rows:
        pf_str = f"{r['pf']:.2f}" if r["pf"] < 999 else "inf"
        print(f"  {r['name']:<28s} {r['trades']:>6d} {r['win_rate']:>7.1f}% {pf_str:>6s} "
              f"{r['cagr']:>+6.2f}% {r['sharpe']:>7.2f} {r['max_dd']:>6.2f}% "
              f"{r['return_pct']:>+7.2f}% {r['avg_hold']:>7.1f}d")
    print("=" * 100)

    # Show detailed results for the best scenario
    if best_result:
        print(f"\n  >>> Best scenario: {best_name}")
        analyze(best_result)


if __name__ == "__main__":
    # Fetch data
    df = fetch_data("GC=F", start="2005-01-01", end="2026-02-10")

    # Run scenario comparison
    run_scenario_comparison(df)

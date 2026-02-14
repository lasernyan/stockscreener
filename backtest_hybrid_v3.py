"""
XAUUSD Hybrid Strategy v3 — Python Backtest
=============================================
Replicates the Pine Script dual-engine strategy:
  Engine 1 (Breakout):     Donchian channel break, tight trailing stop
  Engine 2 (Trend-Follow): EMA crossover + pullback, wide trailing stop
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tabulate import tabulate
import json
import urllib.request
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════

def fetch_xauusd_data():
    """Fetch XAUUSD daily data from multiple free API sources."""

    # Try Yahoo Finance CSV endpoint (no auth required for daily)
    symbols = ["GC=F", "GLD"]  # Gold futures, Gold ETF as fallback
    for symbol in symbols:
        try:
            end_ts = int(datetime.now().timestamp())
            start_ts = int((datetime.now() - timedelta(days=365*5)).timestamp())
            url = (f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                   f"?period1={start_ts}&period2={end_ts}&interval=1d"
                   f"&events=history&includeAdjustedClose=true")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            response = urllib.request.urlopen(req, timeout=15)
            data = response.read().decode('utf-8')
            lines = data.strip().split('\n')
            if len(lines) > 100:
                rows = []
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 6 and parts[1] != 'null':
                        try:
                            rows.append({
                                'Date': parts[0],
                                'Open': float(parts[1]),
                                'High': float(parts[2]),
                                'Low': float(parts[3]),
                                'Close': float(parts[4]),
                                'Volume': float(parts[5]) if parts[5] != 'null' else 0
                            })
                        except ValueError:
                            continue
                if len(rows) > 100:
                    df = pd.DataFrame(rows)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    print(f"  Fetched {len(df)} bars from Yahoo Finance ({symbol})")
                    return df, symbol
        except Exception as e:
            print(f"  Yahoo Finance ({symbol}) failed: {e}")
            continue

    return None, None


def generate_synthetic_xauusd(n_bars=2500):
    """Generate realistic synthetic XAUUSD data using geometric Brownian motion
    calibrated to gold's actual statistical properties."""

    np.random.seed(42)

    # Gold statistical properties (daily)
    start_price = 1300.0     # Start from 2019 level
    annual_drift = 0.08      # ~8% annual return (gold 2019-2024 avg)
    annual_vol = 0.16        # ~16% annualized volatility
    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)

    # Regime switching for realistic volatility clustering
    dates = pd.bdate_range(start='2019-01-02', periods=n_bars, freq='B')
    prices = np.zeros(n_bars)
    prices[0] = start_price

    # Simulate with GARCH-like vol clustering
    vol_state = daily_vol
    for i in range(1, n_bars):
        # Regime shift probability
        if np.random.random() < 0.02:  # 2% chance of vol regime change
            vol_state = daily_vol * np.random.uniform(0.5, 2.5)
        else:
            vol_state = 0.95 * vol_state + 0.05 * daily_vol  # Mean revert

        ret = daily_drift + vol_state * np.random.randn()
        prices[i] = prices[i-1] * (1 + ret)

    # Generate OHLC from close prices
    df = pd.DataFrame(index=dates[:len(prices)])
    df['Close'] = prices

    # Realistic intraday ranges
    daily_range = daily_vol * prices * np.random.uniform(0.3, 1.5, n_bars)
    open_offset = (np.random.randn(n_bars) * 0.3) * daily_range

    df['Open'] = prices + open_offset
    df['High'] = np.maximum(df['Open'], df['Close']) + np.abs(daily_range * np.random.uniform(0.1, 0.6, n_bars))
    df['Low'] = np.minimum(df['Open'], df['Close']) - np.abs(daily_range * np.random.uniform(0.1, 0.6, n_bars))
    df['Volume'] = np.random.lognormal(mean=15, sigma=0.5, size=n_bars)

    df.index.name = 'Date'
    return df


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def calc_atr(df, period=14):
    """Average True Range."""
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
    atr = pd.Series(tr, index=df.index).rolling(period).mean()
    return atr


def calc_ema(series, period):
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calc_adx(df, period=14):
    """ADX via DMI."""
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))

    # Smoothed using Wilder's method
    atr_smooth = pd.Series(tr).rolling(period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / np.where(atr_smooth > 0, atr_smooth, 1)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / np.where(atr_smooth > 0, atr_smooth, 1)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    adx = pd.Series(dx, index=df.index).rolling(period).mean()
    return adx


def calc_donchian(df, period=20):
    """Donchian channel (shifted by 1 to avoid look-ahead)."""
    upper = df['High'].rolling(period).max().shift(1)
    lower = df['Low'].rolling(period).min().shift(1)
    return upper, lower


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class HybridBacktester:
    """Dual-engine backtest: Breakout + Trend-Follow."""

    def __init__(self, df, config=None):
        self.df = df.copy()
        self.c = {
            # Shared
            'initial_capital': 100000,
            'atr_period': 14,
            'adx_period': 14,
            'adx_threshold': 20,
            'adx_require_rise': True,
            'cooldown_bars': 5,
            'max_daily_loss_pct': 3.0,
            'slippage_pct': 0.03,    # 0.03% per trade (~$0.60 on $2000 gold)
            'commission_pct': 0.02,  # 0.02% commission

            # Breakout engine
            'bo_enabled': True,
            'bo_lookback': 20,
            'bo_sl_mult': 1.5,
            'bo_tp_mult': 3.0,
            'bo_use_trail': True,
            'bo_trail_activation': 1.5,
            'bo_trail_distance': 2.0,
            'bo_breakeven_atr': 1.0,
            'bo_momentum_thresh': 0.3,
            'bo_risk_pct': 1.0,

            # Trend-Follow engine
            'tf_enabled': True,
            'tf_fast_ma': 50,
            'tf_slow_ma': 200,
            'tf_pullback_ma': 50,
            'tf_sl_mult': 2.0,
            'tf_use_trail': True,
            'tf_trail_activation': 2.5,
            'tf_trail_distance': 4.0,
            'tf_breakeven_atr': 2.0,
            'tf_tp_mult': 6.0,
            'tf_pullback_atr': 0.5,
            'tf_risk_pct': 1.0,
        }
        if config:
            self.c.update(config)

    def run(self):
        df = self.df
        c = self.c
        n = len(df)

        # Pre-compute indicators
        atr = calc_atr(df, c['atr_period'])
        adx = calc_adx(df, c['adx_period'])
        upper_ch, lower_ch = calc_donchian(df, c['bo_lookback'])
        ema_fast = calc_ema(df['Close'], c['tf_fast_ma'])
        ema_slow = calc_ema(df['Close'], c['tf_slow_ma'])
        ema_pullback = calc_ema(df['Close'], c['tf_pullback_ma'])

        # State
        equity = c['initial_capital']
        equity_curve = np.full(n, np.nan)
        equity_curve[0] = equity

        trades = []       # completed trades
        daily_pnl = 0.0
        last_day = None

        # Per-engine state
        bo_pos = None   # {'direction', 'entry_price', 'qty', 'sl', 'best', 'trail_atr', 'trail_active', 'be_hit', 'entry_bar'}
        tf_pos = None
        bo_cooldown = 0
        tf_cooldown = 0

        close = df['Close'].values
        high_arr = df['High'].values
        low_arr = df['Low'].values
        open_arr = df['Open'].values
        dates = df.index

        warmup = max(c['tf_slow_ma'], c['bo_lookback'], c['atr_period'], c['adx_period']) + 5

        for i in range(warmup, n):
            cur_atr = atr.iloc[i]
            cur_adx = adx.iloc[i]
            cur_close = close[i]
            cur_high = high_arr[i]
            cur_low = low_arr[i]
            cur_open = open_arr[i]
            cur_date = dates[i]

            if pd.isna(cur_atr) or cur_atr <= 0 or pd.isna(cur_adx):
                equity_curve[i] = equity
                continue

            # Daily P&L reset
            cur_day = cur_date.date() if hasattr(cur_date, 'date') else cur_date
            if cur_day != last_day:
                daily_pnl = 0.0
                last_day = cur_day

            daily_loss_limit = daily_pnl <= -(equity * c['max_daily_loss_pct'] / 100)

            # ADX filter
            adx_ok = True
            if cur_adx < c['adx_threshold']:
                adx_ok = False
            if c['adx_require_rise'] and i >= 3:
                if not (cur_adx > adx.iloc[i-3]):
                    adx_ok = False

            # Cooldown
            if bo_cooldown > 0:
                bo_cooldown -= 1
            if tf_cooldown > 0:
                tf_cooldown -= 1

            # ─── CHECK STOPS (at bar open / during bar) ───────────────
            # Simulate: check if SL was hit during the bar

            # BO position stop check
            if bo_pos is not None:
                stopped = False
                if bo_pos['direction'] == 'long':
                    bo_pos['best'] = max(bo_pos['best'], cur_high)
                    # Check SL hit (low touched SL)
                    if cur_low <= bo_pos['sl']:
                        exit_price = bo_pos['sl'] * (1 - c['slippage_pct']/100)
                        pnl = (exit_price - bo_pos['entry_price']) * bo_pos['qty']
                        comm = exit_price * bo_pos['qty'] * c['commission_pct'] / 100
                        net_pnl = pnl - comm
                        equity += net_pnl
                        daily_pnl += net_pnl
                        trades.append({
                            'engine': 'BO', 'direction': 'long',
                            'entry_date': bo_pos['entry_date'], 'exit_date': cur_date,
                            'entry_price': bo_pos['entry_price'], 'exit_price': exit_price,
                            'qty': bo_pos['qty'], 'pnl': net_pnl,
                            'bars_held': i - bo_pos['entry_bar'],
                            'exit_reason': 'trail_sl' if bo_pos['trail_active'] else ('be_sl' if bo_pos['be_hit'] else 'initial_sl')
                        })
                        if net_pnl < 0:
                            bo_cooldown = c['cooldown_bars']
                        bo_pos = None
                        stopped = True
                else:  # short
                    bo_pos['best'] = min(bo_pos['best'], cur_low)
                    if cur_high >= bo_pos['sl']:
                        exit_price = bo_pos['sl'] * (1 + c['slippage_pct']/100)
                        pnl = (bo_pos['entry_price'] - exit_price) * bo_pos['qty']
                        comm = exit_price * bo_pos['qty'] * c['commission_pct'] / 100
                        net_pnl = pnl - comm
                        equity += net_pnl
                        daily_pnl += net_pnl
                        trades.append({
                            'engine': 'BO', 'direction': 'short',
                            'entry_date': bo_pos['entry_date'], 'exit_date': cur_date,
                            'entry_price': bo_pos['entry_price'], 'exit_price': exit_price,
                            'qty': bo_pos['qty'], 'pnl': net_pnl,
                            'bars_held': i - bo_pos['entry_bar'],
                            'exit_reason': 'trail_sl' if bo_pos['trail_active'] else ('be_sl' if bo_pos['be_hit'] else 'initial_sl')
                        })
                        if net_pnl < 0:
                            bo_cooldown = c['cooldown_bars']
                        bo_pos = None
                        stopped = True

                # Update trailing / breakeven if not stopped
                if not stopped and bo_pos is not None:
                    atr_at_entry = bo_pos['trail_atr']
                    if bo_pos['direction'] == 'long':
                        # Breakeven
                        if not bo_pos['be_hit'] and bo_pos['best'] >= bo_pos['entry_price'] + c['bo_breakeven_atr'] * atr_at_entry:
                            bo_pos['sl'] = max(bo_pos['sl'], bo_pos['entry_price'])
                            bo_pos['be_hit'] = True
                        # Trailing
                        if c['bo_use_trail']:
                            if bo_pos['trail_active']:
                                trail_sl = bo_pos['best'] - c['bo_trail_distance'] * atr_at_entry
                                bo_pos['sl'] = max(bo_pos['sl'], trail_sl)
                            elif bo_pos['best'] >= bo_pos['entry_price'] + c['bo_trail_activation'] * atr_at_entry:
                                bo_pos['trail_active'] = True
                                trail_sl = bo_pos['best'] - c['bo_trail_distance'] * atr_at_entry
                                bo_pos['sl'] = max(bo_pos['sl'], trail_sl)
                    else:  # short
                        if not bo_pos['be_hit'] and bo_pos['best'] <= bo_pos['entry_price'] - c['bo_breakeven_atr'] * atr_at_entry:
                            bo_pos['sl'] = min(bo_pos['sl'], bo_pos['entry_price'])
                            bo_pos['be_hit'] = True
                        if c['bo_use_trail']:
                            if bo_pos['trail_active']:
                                trail_sl = bo_pos['best'] + c['bo_trail_distance'] * atr_at_entry
                                bo_pos['sl'] = min(bo_pos['sl'], trail_sl)
                            elif bo_pos['best'] <= bo_pos['entry_price'] - c['bo_trail_activation'] * atr_at_entry:
                                bo_pos['trail_active'] = True
                                trail_sl = bo_pos['best'] + c['bo_trail_distance'] * atr_at_entry
                                bo_pos['sl'] = min(bo_pos['sl'], trail_sl)

            # TF position stop check
            if tf_pos is not None:
                stopped = False
                if tf_pos['direction'] == 'long':
                    tf_pos['best'] = max(tf_pos['best'], cur_high)
                    if cur_low <= tf_pos['sl']:
                        exit_price = tf_pos['sl'] * (1 - c['slippage_pct']/100)
                        pnl = (exit_price - tf_pos['entry_price']) * tf_pos['qty']
                        comm = exit_price * tf_pos['qty'] * c['commission_pct'] / 100
                        net_pnl = pnl - comm
                        equity += net_pnl
                        daily_pnl += net_pnl
                        trades.append({
                            'engine': 'TF', 'direction': 'long',
                            'entry_date': tf_pos['entry_date'], 'exit_date': cur_date,
                            'entry_price': tf_pos['entry_price'], 'exit_price': exit_price,
                            'qty': tf_pos['qty'], 'pnl': net_pnl,
                            'bars_held': i - tf_pos['entry_bar'],
                            'exit_reason': 'trail_sl' if tf_pos['trail_active'] else ('be_sl' if tf_pos['be_hit'] else 'initial_sl')
                        })
                        if net_pnl < 0:
                            tf_cooldown = c['cooldown_bars']
                        tf_pos = None
                        stopped = True
                else:  # short
                    tf_pos['best'] = min(tf_pos['best'], cur_low)
                    if cur_high >= tf_pos['sl']:
                        exit_price = tf_pos['sl'] * (1 + c['slippage_pct']/100)
                        pnl = (tf_pos['entry_price'] - exit_price) * tf_pos['qty']
                        comm = exit_price * tf_pos['qty'] * c['commission_pct'] / 100
                        net_pnl = pnl - comm
                        equity += net_pnl
                        daily_pnl += net_pnl
                        trades.append({
                            'engine': 'TF', 'direction': 'short',
                            'entry_date': tf_pos['entry_date'], 'exit_date': cur_date,
                            'entry_price': tf_pos['entry_price'], 'exit_price': exit_price,
                            'qty': tf_pos['qty'], 'pnl': net_pnl,
                            'bars_held': i - tf_pos['entry_bar'],
                            'exit_reason': 'trail_sl' if tf_pos['trail_active'] else ('be_sl' if tf_pos['be_hit'] else 'initial_sl')
                        })
                        if net_pnl < 0:
                            tf_cooldown = c['cooldown_bars']
                        tf_pos = None
                        stopped = True

                if not stopped and tf_pos is not None:
                    atr_at_entry = tf_pos['trail_atr']
                    trend_up = ema_fast.iloc[i] > ema_slow.iloc[i]
                    trend_down = ema_fast.iloc[i] < ema_slow.iloc[i]

                    if tf_pos['direction'] == 'long':
                        if not tf_pos['be_hit'] and tf_pos['best'] >= tf_pos['entry_price'] + c['tf_breakeven_atr'] * atr_at_entry:
                            tf_pos['sl'] = max(tf_pos['sl'], tf_pos['entry_price'])
                            tf_pos['be_hit'] = True
                        if c['tf_use_trail']:
                            if tf_pos['trail_active']:
                                trail_sl = tf_pos['best'] - c['tf_trail_distance'] * atr_at_entry
                                tf_pos['sl'] = max(tf_pos['sl'], trail_sl)
                            elif tf_pos['best'] >= tf_pos['entry_price'] + c['tf_trail_activation'] * atr_at_entry:
                                tf_pos['trail_active'] = True
                                trail_sl = tf_pos['best'] - c['tf_trail_distance'] * atr_at_entry
                                tf_pos['sl'] = max(tf_pos['sl'], trail_sl)
                        # Trend invalidation
                        if not trend_up:
                            emergency_sl = tf_pos['best'] - c['bo_trail_distance'] * atr_at_entry
                            tf_pos['sl'] = max(tf_pos['sl'], emergency_sl)
                    else:
                        if not tf_pos['be_hit'] and tf_pos['best'] <= tf_pos['entry_price'] - c['tf_breakeven_atr'] * atr_at_entry:
                            tf_pos['sl'] = min(tf_pos['sl'], tf_pos['entry_price'])
                            tf_pos['be_hit'] = True
                        if c['tf_use_trail']:
                            if tf_pos['trail_active']:
                                trail_sl = tf_pos['best'] + c['tf_trail_distance'] * atr_at_entry
                                tf_pos['sl'] = min(tf_pos['sl'], trail_sl)
                            elif tf_pos['best'] <= tf_pos['entry_price'] - c['tf_trail_activation'] * atr_at_entry:
                                tf_pos['trail_active'] = True
                                trail_sl = tf_pos['best'] + c['tf_trail_distance'] * atr_at_entry
                                tf_pos['sl'] = min(tf_pos['sl'], trail_sl)
                        if not trend_down:
                            emergency_sl = tf_pos['best'] + c['bo_trail_distance'] * atr_at_entry
                            tf_pos['sl'] = min(tf_pos['sl'], emergency_sl)

            # ─── ENTRY SIGNALS ────────────────────────────────────────

            # Breakout signals
            if c['bo_enabled'] and bo_pos is None and bo_cooldown == 0 and not daily_loss_limit and adx_ok:
                uc = upper_ch.iloc[i]
                lc = lower_ch.iloc[i]
                if not pd.isna(uc) and not pd.isna(lc):
                    # Long breakout
                    if cur_close > uc:
                        strength = (cur_close - uc) / cur_atr
                        if strength > c['bo_momentum_thresh']:
                            entry_price = cur_close * (1 + c['slippage_pct']/100)
                            sl = entry_price - c['bo_sl_mult'] * cur_atr
                            risk_amt = equity * c['bo_risk_pct'] / 100
                            sl_dist = c['bo_sl_mult'] * cur_atr
                            qty = risk_amt / sl_dist if sl_dist > 0 else 0
                            if qty > 0:
                                comm = entry_price * qty * c['commission_pct'] / 100
                                equity -= comm
                                bo_pos = {
                                    'direction': 'long', 'entry_price': entry_price,
                                    'qty': qty, 'sl': sl, 'best': entry_price,
                                    'trail_atr': cur_atr, 'trail_active': False,
                                    'be_hit': False, 'entry_bar': i, 'entry_date': cur_date
                                }

                    # Short breakout
                    elif cur_close < lc:
                        strength = (lc - cur_close) / cur_atr
                        if strength > c['bo_momentum_thresh']:
                            entry_price = cur_close * (1 - c['slippage_pct']/100)
                            sl = entry_price + c['bo_sl_mult'] * cur_atr
                            risk_amt = equity * c['bo_risk_pct'] / 100
                            sl_dist = c['bo_sl_mult'] * cur_atr
                            qty = risk_amt / sl_dist if sl_dist > 0 else 0
                            if qty > 0:
                                comm = entry_price * qty * c['commission_pct'] / 100
                                equity -= comm
                                bo_pos = {
                                    'direction': 'short', 'entry_price': entry_price,
                                    'qty': qty, 'sl': sl, 'best': entry_price,
                                    'trail_atr': cur_atr, 'trail_active': False,
                                    'be_hit': False, 'entry_bar': i, 'entry_date': cur_date
                                }

            # Trend-Follow signals
            if c['tf_enabled'] and tf_pos is None and tf_cooldown == 0 and not daily_loss_limit and adx_ok:
                ef = ema_fast.iloc[i]
                es = ema_slow.iloc[i]
                ep = ema_pullback.iloc[i]

                if not pd.isna(ef) and not pd.isna(es) and not pd.isna(ep) and i >= 1:
                    trend_up = ef > es and ema_fast.iloc[i-1] > ema_slow.iloc[i-1]
                    trend_down = ef < es and ema_fast.iloc[i-1] < ema_slow.iloc[i-1]

                    # Long: uptrend + pullback to EMA + bounce
                    if trend_up and cur_close > es:
                        pullback_touch = cur_low <= ep + c['tf_pullback_atr'] * cur_atr
                        bounce = cur_close > cur_open and cur_close > ep
                        if pullback_touch and bounce:
                            entry_price = cur_close * (1 + c['slippage_pct']/100)
                            sl = entry_price - c['tf_sl_mult'] * cur_atr
                            risk_amt = equity * c['tf_risk_pct'] / 100
                            sl_dist = c['tf_sl_mult'] * cur_atr
                            qty = risk_amt / sl_dist if sl_dist > 0 else 0
                            if qty > 0:
                                comm = entry_price * qty * c['commission_pct'] / 100
                                equity -= comm
                                tf_pos = {
                                    'direction': 'long', 'entry_price': entry_price,
                                    'qty': qty, 'sl': sl, 'best': entry_price,
                                    'trail_atr': cur_atr, 'trail_active': False,
                                    'be_hit': False, 'entry_bar': i, 'entry_date': cur_date
                                }

                    # Short: downtrend + pullback to EMA + drop
                    elif trend_down and cur_close < es:
                        pullback_touch = cur_high >= ep - c['tf_pullback_atr'] * cur_atr
                        bounce = cur_close < cur_open and cur_close < ep
                        if pullback_touch and bounce:
                            entry_price = cur_close * (1 - c['slippage_pct']/100)
                            sl = entry_price + c['tf_sl_mult'] * cur_atr
                            risk_amt = equity * c['tf_risk_pct'] / 100
                            sl_dist = c['tf_sl_mult'] * cur_atr
                            qty = risk_amt / sl_dist if sl_dist > 0 else 0
                            if qty > 0:
                                comm = entry_price * qty * c['commission_pct'] / 100
                                equity -= comm
                                tf_pos = {
                                    'direction': 'short', 'entry_price': entry_price,
                                    'qty': qty, 'sl': sl, 'best': entry_price,
                                    'trail_atr': cur_atr, 'trail_active': False,
                                    'be_hit': False, 'entry_bar': i, 'entry_date': cur_date
                                }

            # Mark-to-market equity
            mtm = 0.0
            if bo_pos is not None:
                if bo_pos['direction'] == 'long':
                    mtm += (cur_close - bo_pos['entry_price']) * bo_pos['qty']
                else:
                    mtm += (bo_pos['entry_price'] - cur_close) * bo_pos['qty']
            if tf_pos is not None:
                if tf_pos['direction'] == 'long':
                    mtm += (cur_close - tf_pos['entry_price']) * tf_pos['qty']
                else:
                    mtm += (tf_pos['entry_price'] - cur_close) * tf_pos['qty']

            equity_curve[i] = equity + mtm

        # Close any remaining positions at last bar
        last_close = close[-1]
        for pos, engine_name in [(bo_pos, 'BO'), (tf_pos, 'TF')]:
            if pos is not None:
                if pos['direction'] == 'long':
                    exit_price = last_close * (1 - c['slippage_pct']/100)
                    pnl = (exit_price - pos['entry_price']) * pos['qty']
                else:
                    exit_price = last_close * (1 + c['slippage_pct']/100)
                    pnl = (pos['entry_price'] - exit_price) * pos['qty']
                comm = exit_price * pos['qty'] * c['commission_pct'] / 100
                net_pnl = pnl - comm
                equity += net_pnl
                trades.append({
                    'engine': engine_name, 'direction': pos['direction'],
                    'entry_date': pos['entry_date'], 'exit_date': dates[-1],
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'qty': pos['qty'], 'pnl': net_pnl,
                    'bars_held': n - 1 - pos['entry_bar'],
                    'exit_reason': 'end_of_data'
                })

        equity_curve[-1] = equity

        # Forward-fill equity curve
        eq_series = pd.Series(equity_curve, index=df.index)
        eq_series = eq_series.ffill()
        eq_series = eq_series.fillna(c['initial_capital'])

        return eq_series, trades


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def calc_performance(equity_curve, trades, initial_capital, label="Strategy"):
    """Calculate comprehensive performance metrics."""
    eq = equity_curve.dropna()
    if len(eq) < 2:
        return {}

    total_return = (eq.iloc[-1] / initial_capital - 1) * 100
    n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    cagr = ((eq.iloc[-1] / initial_capital) ** (1 / n_years) - 1) * 100

    # Drawdown
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # Daily returns
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    sortino_denom = daily_ret[daily_ret < 0].std()
    sortino = (daily_ret.mean() / sortino_denom * np.sqrt(252)) if sortino_denom > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t['pnl'] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        win_rate = len(winners) / n_trades * 100
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        profit_factor = abs(sum(winners) / sum(losers)) if sum(losers) != 0 else float('inf')
        avg_bars = np.mean([t['bars_held'] for t in trades])
        expectancy = np.mean(pnls)

        # Max consecutive wins/losses
        streak_w, streak_l, cur_w, cur_l = 0, 0, 0, 0
        for p in pnls:
            if p > 0:
                cur_w += 1; cur_l = 0
            else:
                cur_l += 1; cur_w = 0
            streak_w = max(streak_w, cur_w)
            streak_l = max(streak_l, cur_l)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_bars = expectancy = 0
        streak_w = streak_l = 0

    return {
        'label': label,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'avg_bars_held': avg_bars,
        'max_consec_wins': streak_w,
        'max_consec_losses': streak_l,
        'final_equity': eq.iloc[-1],
        'n_years': n_years,
    }


def calc_engine_stats(trades, engine_code):
    """Calculate per-engine trade statistics."""
    engine_trades = [t for t in trades if t['engine'] == engine_code]
    if not engine_trades:
        return None
    pnls = [t['pnl'] for t in engine_trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    return {
        'trades': len(engine_trades),
        'win_rate': len(winners) / len(engine_trades) * 100 if engine_trades else 0,
        'total_pnl': sum(pnls),
        'avg_pnl': np.mean(pnls),
        'profit_factor': abs(sum(winners) / sum(losers)) if sum(losers) != 0 else float('inf'),
        'avg_bars': np.mean([t['bars_held'] for t in engine_trades]),
        'longs': len([t for t in engine_trades if t['direction'] == 'long']),
        'shorts': len([t for t in engine_trades if t['direction'] == 'short']),
    }


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(df, equity_hybrid, equity_bo, equity_tf, trades, output_dir):
    """Generate comprehensive backtest charts."""

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 2, 1.5, 1.5]})
    fig.suptitle('XAUUSD Hybrid Strategy v3 — Backtest Results', fontsize=16, fontweight='bold')

    # ── Panel 1: Price + Trades ───────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], color='#666666', linewidth=0.8, label='XAUUSD', alpha=0.8)

    # Plot EMA
    ema50 = calc_ema(df['Close'], 50)
    ema200 = calc_ema(df['Close'], 200)
    ax1.plot(df.index, ema50, color='cyan', linewidth=0.7, alpha=0.5, label='EMA 50')
    ax1.plot(df.index, ema200, color='orange', linewidth=0.7, alpha=0.5, label='EMA 200')

    # Plot trades
    for t in trades:
        color = '#00cc00' if t['pnl'] > 0 else '#cc0000'
        marker_entry = '^' if t['direction'] == 'long' else 'v'
        engine_marker = 'o' if t['engine'] == 'BO' else 'D'
        ax1.plot(t['entry_date'], t['entry_price'], marker=marker_entry,
                color=color, markersize=5, alpha=0.7, zorder=5)
        ax1.plot(t['exit_date'], t['exit_price'], marker=engine_marker,
                color=color, markersize=4, alpha=0.5, zorder=5)

    ax1.set_title('Price Chart with Trade Entries/Exits')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Equity Curves ────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(equity_hybrid.index, equity_hybrid.values, color='#00aaff', linewidth=1.5, label='Hybrid (BO+TF)')
    if equity_bo is not None:
        ax2.plot(equity_bo.index, equity_bo.values, color='#00cc00', linewidth=1, alpha=0.7, label='Breakout Only')
    if equity_tf is not None:
        ax2.plot(equity_tf.index, equity_tf.values, color='#ff6600', linewidth=1, alpha=0.7, label='Trend-Follow Only')
    ax2.axhline(y=100000, color='gray', linestyle='--', alpha=0.3)
    ax2.set_title('Equity Curves')
    ax2.set_ylabel('Equity ($)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────
    ax3 = axes[2]
    rolling_max = equity_hybrid.cummax()
    dd = (equity_hybrid - rolling_max) / rolling_max * 100
    ax3.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
    ax3.plot(dd.index, dd.values, color='red', linewidth=0.8)
    ax3.set_title('Drawdown (%)')
    ax3.set_ylabel('Drawdown %')
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: Trade P&L Distribution ───────────────────────────────
    ax4 = axes[3]
    if trades:
        bo_pnls = [t['pnl'] for t in trades if t['engine'] == 'BO']
        tf_pnls = [t['pnl'] for t in trades if t['engine'] == 'TF']
        bins = 30
        if bo_pnls:
            ax4.hist(bo_pnls, bins=bins, alpha=0.5, color='green', label=f'BO ({len(bo_pnls)} trades)')
        if tf_pnls:
            ax4.hist(tf_pnls, bins=bins, alpha=0.5, color='orange', label=f'TF ({len(tf_pnls)} trades)')
        ax4.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax4.set_title('Trade P&L Distribution by Engine')
    ax4.set_xlabel('P&L ($)')
    ax4.set_ylabel('Count')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'backtest_results.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"\n  Chart saved: {chart_path}")
    return chart_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    output_dir = '/home/user/desktop-tutorial'

    print("=" * 70)
    print("  XAUUSD HYBRID STRATEGY v3 — BACKTEST")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df, symbol = fetch_xauusd_data()
    if df is None or len(df) < 300:
        print("  Live data unavailable. Generating synthetic XAUUSD data...")
        print("  (Calibrated to gold's statistical properties: ~8% drift, ~16% vol)")
        df = generate_synthetic_xauusd(2500)
        symbol = "XAUUSD (Synthetic)"
    print(f"  Data: {symbol}")
    print(f"  Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Bars: {len(df)}")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    # ── Run backtests ─────────────────────────────────────────────────
    print("\n[2/4] Running backtests...")

    # Hybrid (both engines)
    print("  Running: Hybrid (Breakout + Trend-Follow)...")
    bt_hybrid = HybridBacktester(df)
    eq_hybrid, trades_hybrid = bt_hybrid.run()

    # Breakout only
    print("  Running: Breakout Only...")
    bt_bo = HybridBacktester(df, {'tf_enabled': False})
    eq_bo, trades_bo = bt_bo.run()

    # Trend-Follow only
    print("  Running: Trend-Follow Only...")
    bt_tf = HybridBacktester(df, {'bo_enabled': False})
    eq_tf, trades_tf = bt_tf.run()

    # ── Performance Metrics ───────────────────────────────────────────
    print("\n[3/4] Calculating performance metrics...")
    perf_hybrid = calc_performance(eq_hybrid, trades_hybrid, 100000, "Hybrid (BO+TF)")
    perf_bo = calc_performance(eq_bo, trades_bo, 100000, "Breakout Only")
    perf_tf = calc_performance(eq_tf, trades_tf, 100000, "Trend-Follow Only")

    # Buy & hold benchmark
    bh_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    bh_equity = 100000 * (df['Close'] / df['Close'].iloc[0])
    perf_bh = calc_performance(bh_equity, [], 100000, "Buy & Hold")

    # ── Print Results ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PERFORMANCE COMPARISON")
    print("=" * 70)

    comparison = []
    for p in [perf_hybrid, perf_bo, perf_tf, perf_bh]:
        if p:
            comparison.append([
                p.get('label', ''),
                f"{p.get('total_return', 0):.1f}%",
                f"{p.get('cagr', 0):.1f}%",
                f"{p.get('max_drawdown', 0):.1f}%",
                f"{p.get('sharpe', 0):.2f}",
                f"{p.get('sortino', 0):.2f}",
                f"{p.get('calmar', 0):.2f}",
                f"{p.get('n_trades', 0)}",
                f"{p.get('win_rate', 0):.1f}%",
                f"{p.get('profit_factor', 0):.2f}",
                f"${p.get('final_equity', 0):,.0f}",
            ])

    headers = ['Strategy', 'Total Ret', 'CAGR', 'Max DD', 'Sharpe', 'Sortino',
               'Calmar', 'Trades', 'Win%', 'PF', 'Final Equity']
    print(tabulate(comparison, headers=headers, tablefmt='grid'))

    # Per-engine breakdown
    print("\n" + "=" * 70)
    print("  ENGINE BREAKDOWN (Hybrid Strategy)")
    print("=" * 70)

    bo_stats = calc_engine_stats(trades_hybrid, 'BO')
    tf_stats = calc_engine_stats(trades_hybrid, 'TF')

    engine_table = []
    for name, stats in [("Breakout", bo_stats), ("Trend-Follow", tf_stats)]:
        if stats:
            engine_table.append([
                name,
                stats['trades'],
                f"{stats['win_rate']:.1f}%",
                f"${stats['total_pnl']:,.0f}",
                f"${stats['avg_pnl']:,.0f}",
                f"{stats['profit_factor']:.2f}",
                f"{stats['avg_bars']:.1f}",
                f"{stats['longs']}L / {stats['shorts']}S",
            ])

    engine_headers = ['Engine', 'Trades', 'Win%', 'Total P&L', 'Avg P&L', 'PF', 'Avg Bars', 'L/S Split']
    print(tabulate(engine_table, headers=engine_headers, tablefmt='grid'))

    # Exit reason breakdown
    print("\n  Exit Reasons:")
    exit_reasons = {}
    for t in trades_hybrid:
        r = t['exit_reason']
        if r not in exit_reasons:
            exit_reasons[r] = {'count': 0, 'pnl': 0}
        exit_reasons[r]['count'] += 1
        exit_reasons[r]['pnl'] += t['pnl']

    reason_table = [[r, d['count'], f"${d['pnl']:,.0f}"] for r, d in sorted(exit_reasons.items())]
    print(tabulate(reason_table, headers=['Reason', 'Count', 'Total P&L'], tablefmt='simple'))

    # Top/bottom trades
    if trades_hybrid:
        sorted_trades = sorted(trades_hybrid, key=lambda x: x['pnl'])
        print("\n  Worst 5 Trades:")
        for t in sorted_trades[:5]:
            print(f"    {t['engine']} {t['direction']:5s} | {str(t['entry_date'])[:10]} → {str(t['exit_date'])[:10]} | "
                  f"P&L: ${t['pnl']:,.0f} | Bars: {t['bars_held']} | {t['exit_reason']}")
        print("\n  Best 5 Trades:")
        for t in sorted_trades[-5:]:
            print(f"    {t['engine']} {t['direction']:5s} | {str(t['entry_date'])[:10]} → {str(t['exit_date'])[:10]} | "
                  f"P&L: ${t['pnl']:,.0f} | Bars: {t['bars_held']} | {t['exit_reason']}")

    # Monthly returns
    print("\n" + "=" * 70)
    print("  MONTHLY RETURNS (Hybrid)")
    print("=" * 70)
    monthly_eq = eq_hybrid.resample('ME').last()
    monthly_ret = monthly_eq.pct_change().dropna() * 100
    if len(monthly_ret) > 0:
        years = sorted(set(monthly_ret.index.year))
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_table = []
        for yr in years:
            row = [yr]
            yr_data = monthly_ret[monthly_ret.index.year == yr]
            yr_total = 0
            for m in range(1, 13):
                m_data = yr_data[yr_data.index.month == m]
                if len(m_data) > 0:
                    val = m_data.iloc[0]
                    yr_total += val
                    row.append(f"{val:+.1f}")
                else:
                    row.append("")
            row.append(f"{yr_total:+.1f}")
            monthly_table.append(row)
        print(tabulate(monthly_table, headers=['Year'] + month_names + ['Total'], tablefmt='simple'))

    # ── Charts ────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating charts...")
    try:
        plt.style.use('dark_background')
        chart_path = plot_results(df, eq_hybrid, eq_bo, eq_tf, trades_hybrid, output_dir)
    except Exception as e:
        print(f"  Chart generation failed: {e}")
        chart_path = None

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY & NOTES")
    print("=" * 70)
    print(f"""
  Data Source:     {symbol}
  Backtest Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({perf_hybrid.get('n_years',0):.1f} years)
  Initial Capital: $100,000
  Slippage:        0.03% per fill (~$0.60 on $2,000 gold)
  Commission:      0.02% per fill

  IMPORTANT CAVEATS:
  - Daily timeframe only (Pine Script session filter not applicable)
  - No intrabar simulation (SL checked against bar High/Low only)
  - Synthetic data preserves statistical properties but not regime transitions
  - Real gold has news-driven gaps that synthetic data cannot model
  - For production use, validate on TradingView with real XAUUSD data
    """)

    print("=" * 70)
    print("  BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

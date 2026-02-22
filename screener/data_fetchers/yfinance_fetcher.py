"""
yfinance Data Fetcher
======================
株価・財務データを yfinance から取得し、テクニカル指標を計算する。
既存の sector_flow_analyzer.py と互換性のある DataFrame 形式で返す。

対応市場:
    JP — 東証上場銘柄 (ticker suffix: .T)
    US — 米国市場 (NYSE/NASDAQ)

テクニカル指標:
    RSI(14), MACD(12,26,9), ADX(14), SMA20/50/200
    Relative Volume (20日平均比)

Usage:
    fetcher = YFinanceFetcher()
    df_jp = fetcher.fetch_market_snapshot("JP", top_n=200)
    df_us = fetcher.fetch_market_snapshot("US", top_n=200)
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 銘柄リスト: セクター別主要銘柄 (TOPIX 分類ベース)
# ---------------------------------------------------------------------------

JP_SECTOR_TICKERS: dict[str, list[str]] = {
    "エネルギー資源": [
        "5020.T",   # ENEOSホールディングス
        "5019.T",   # 出光興産
        "5021.T",   # コスモエネルギーホールディングス
        "1605.T",   # INPEX
        "1662.T",   # 石油資源開発
        "9531.T",   # 東京ガス
        "9532.T",   # 大阪ガス
    ],
    "素材・化学": [
        "4063.T",   # 信越化学工業
        "3402.T",   # 東レ
        "4005.T",   # 住友化学
        "4183.T",   # 三井化学
        "4042.T",   # 東ソー
        "4188.T",   # 三菱ケミカルグループ
        "4901.T",   # 富士フイルムホールディングス
    ],
    "鉄鋼・非鉄": [
        "5401.T",   # 日本製鉄
        "5411.T",   # JFEホールディングス
        "5802.T",   # 住友電気工業
        "5713.T",   # 住友金属鉱山
        "5406.T",   # 神戸製鋼所
        "5703.T",   # 日本軽金属ホールディングス
    ],
    "建設・資材": [
        "1803.T",   # 清水建設
        "1801.T",   # 大成建設
        "1802.T",   # 大林組
        "1925.T",   # 大和ハウス工業
        "1928.T",   # 積水ハウス
        "5233.T",   # 太平洋セメント
        "3405.T",   # クラレ
    ],
    "機械・資本財": [
        "6301.T",   # コマツ
        "6326.T",   # クボタ
        "7011.T",   # 三菱重工業
        "7012.T",   # 川崎重工業
        "6366.T",   # 千代田化工建設
        "6113.T",   # アマダ
        "6361.T",   # 荏原製作所
    ],
    "電機・精密機器": [
        "6954.T",   # ファナック
        "6273.T",   # SMC
        "6861.T",   # キーエンス
        "6481.T",   # THK
        "6645.T",   # オムロン
        "6981.T",   # 村田製作所
        "6857.T",   # アドバンテスト
        "8035.T",   # 東京エレクトロン
    ],
    "電機・家電": [
        "6758.T",   # ソニーグループ
        "6752.T",   # パナソニックホールディングス
        "6971.T",   # 京セラ
        "6753.T",   # シャープ
        "6502.T",   # 東芝
        "6501.T",   # 日立製作所
    ],
    "情報通信": [
        "9432.T",   # 日本電信電話(NTT)
        "9433.T",   # KDDI
        "9434.T",   # ソフトバンク
        "9984.T",   # ソフトバンクグループ
        "6702.T",   # 富士通
        "6701.T",   # NEC
        "4307.T",   # 野村総合研究所
        "4684.T",   # オービック
        "3659.T",   # ネクソン
    ],
    "ヘルスケア・医薬品": [
        "4502.T",   # 武田薬品工業
        "4503.T",   # アステラス製薬
        "4568.T",   # 第一三共
        "4519.T",   # 中外製薬
        "4523.T",   # エーザイ
        "4543.T",   # テルモ
        "4151.T",   # 協和キリン
        "4530.T",   # 久光製薬
    ],
    "自動車・輸送機器": [
        "7203.T",   # トヨタ自動車
        "7267.T",   # 本田技研工業
        "7201.T",   # 日産自動車
        "7261.T",   # マツダ
        "7270.T",   # SUBARU
        "7272.T",   # ヤマハ発動機
        "7269.T",   # スズキ
        "7013.T",   # IHI
    ],
    "商社・卸売": [
        "8058.T",   # 三菱商事
        "8031.T",   # 三井物産
        "8001.T",   # 伊藤忠商事
        "8053.T",   # 住友商事
        "8002.T",   # 丸紅
        "9001.T",   # 東武鉄道
        "8015.T",   # 豊田通商
    ],
    "金融・銀行": [
        "8306.T",   # 三菱UFJフィナンシャル・グループ
        "8316.T",   # 三井住友フィナンシャルグループ
        "8411.T",   # みずほフィナンシャルグループ
        "8601.T",   # 大和証券グループ本社
        "8604.T",   # 野村ホールディングス
        "8766.T",   # 東京海上ホールディングス
        "8725.T",   # MS&ADインシュアランスグループホールディングス
        "8750.T",   # 第一生命ホールディングス
    ],
    "不動産": [
        "8801.T",   # 三井不動産
        "8802.T",   # 三菱地所
        "8830.T",   # 住友不動産
        "3003.T",   # ヒューリック
        "3289.T",   # 東急不動産ホールディングス
        "8804.T",   # 東京建物
        "3231.T",   # 野村不動産ホールディングス
    ],
    "公益・電力": [
        "9501.T",   # 東京電力ホールディングス
        "9503.T",   # 関西電力
        "9502.T",   # 中部電力
        "9504.T",   # 中国電力
        "9507.T",   # 四国電力
        "9508.T",   # 九州電力
    ],
    "小売・消費": [
        "9983.T",   # ファーストリテイリング(ユニクロ)
        "3382.T",   # セブン&アイ・ホールディングス
        "9843.T",   # ニトリホールディングス
        "2651.T",   # ローソン
        "7453.T",   # 良品計画(無印良品)
        "3099.T",   # 三越伊勢丹ホールディングス
        "8273.T",   # イズミ
    ],
    "食品・生活必需品": [
        "2802.T",   # 味の素
        "2503.T",   # キリンホールディングス
        "2502.T",   # アサヒグループホールディングス
        "4911.T",   # 資生堂
        "2801.T",   # キッコーマン
        "2267.T",   # ヤクルト本社
        "2914.T",   # JT(日本たばこ産業)
    ],
    "運輸・物流": [
        "9020.T",   # 東日本旅客鉄道(JR東日本)
        "9022.T",   # 東海旅客鉄道(JR東海)
        "9021.T",   # 西日本旅客鉄道(JR西日本)
        "9101.T",   # 日本郵船
        "9104.T",   # 商船三井
        "9107.T",   # 川崎汽船
        "9064.T",   # ヤマトホールディングス
    ],
}

US_SECTOR_TICKERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN",
        "AMAT", "LRCX", "MU", "ADI", "KLAC", "MCHP", "NXPI",
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN",
        "MDT", "ABT", "TMO", "DHR", "SYK", "ELV", "CI",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BRK-B", "V", "MA",
        "AXP", "BLK", "C", "USB", "PNC", "TFC", "COF",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW",
        "BKNG", "GM", "F", "ORLY", "AZO", "ROST", "TJX",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KMB",
        "GIS", "CAG", "MO", "PM", "KHC", "STZ", "CHD",
    ],
    "Energy": [
        "XOM", "CVX", "SLB", "EOG", "COP", "PXD", "MPC", "PSX",
        "VLO", "OXY", "HAL", "DVN", "HES", "FANG", "APA",
    ],
    "Materials": [
        "LIN", "APD", "DD", "DOW", "NEM", "FCX", "ALB", "CF",
        "MOS", "PPG", "ECL", "SHW", "VMC", "MLM", "PKG",
    ],
    "Industrials": [
        "UPS", "CAT", "GE", "MMM", "HON", "LMT", "RTX", "BA",
        "DE", "EMR", "NSC", "UNP", "CSX", "FDX", "WM",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "AEP", "EXC", "D", "PCG", "XEL",
        "WEC", "ES", "ETR", "FE", "PPL", "CMS", "AES",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "DLR", "O",
        "VICI", "AVB", "EQR", "WY", "ARE", "BXP", "INVH",
    ],
    "Communication Services": [
        "META", "GOOGL", "NFLX", "DIS", "VZ", "T", "CMCSA",
        "TMUS", "EA", "WBD", "PARA", "FOXA", "OMC", "IPG",
    ],
}


# ---------------------------------------------------------------------------
# Technical Indicator Functions
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(14) を計算する"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """MACD ライン と シグナルライン を計算する"""
    exp_fast = close.ewm(span=fast, adjust=False).mean()
    exp_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp_fast - exp_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX(14) を計算する"""
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder's Smoothing (EWM with alpha=1/period)
    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan))

    di_diff = (plus_di - minus_di).abs()
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (di_diff / di_sum)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx


def _compute_indicators(hist: pd.DataFrame) -> dict:
    """
    価格履歴 DataFrame からテクニカル指標を計算し、
    最新値の辞書を返す。
    """
    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    volume = hist["Volume"].astype(float)

    n = len(close)

    # RSI
    rsi_series = _compute_rsi(close)
    rsi = float(rsi_series.iloc[-1]) if n >= 15 else float("nan")

    # MACD
    if n >= 27:
        macd_line, signal_line = _compute_macd(close)
        macd_val = float(macd_line.iloc[-1])
        macd_sig = float(signal_line.iloc[-1])
    else:
        macd_val = macd_sig = float("nan")

    # ADX
    adx_val = float(_compute_adx(high, low, close).iloc[-1]) if n >= 20 else float("nan")

    # SMA
    sma20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else float("nan")
    sma50 = float(close.rolling(50).mean().iloc[-1]) if n >= 50 else float("nan")
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else float("nan")

    # Relative Volume (20-day avg)
    avg_vol_20 = float(volume.rolling(20).mean().iloc[-1]) if n >= 20 else float(volume.mean())
    current_vol = float(volume.iloc[-1])
    rel_vol = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # Change %
    current_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if n >= 2 else current_close
    change_pct = (current_close - prev_close) / prev_close * 100.0 if prev_close != 0 else 0.0

    return {
        "Close": current_close,
        "Change %": round(change_pct, 4),
        "Volume": current_vol,
        "Relative Volume": round(rel_vol, 4),
        "RSI": round(rsi, 4) if not np.isnan(rsi) else None,
        "MACD.macd": round(macd_val, 6) if not np.isnan(macd_val) else None,
        "MACD.signal": round(macd_sig, 6) if not np.isnan(macd_sig) else None,
        "ADX": round(adx_val, 4) if not np.isnan(adx_val) else None,
        "SMA20": round(sma20, 4) if not np.isnan(sma20) else None,
        "SMA50": round(sma50, 4) if not np.isnan(sma50) else None,
        "SMA200": round(sma200, 4) if not np.isnan(sma200) else None,
    }


# ---------------------------------------------------------------------------
# Main Fetcher Class
# ---------------------------------------------------------------------------

class YFinanceFetcher:
    """
    yfinance を使って株価・テクニカル指標を取得するクラス。

    Parameters
    ----------
    period_days : int
        取得する価格履歴の日数 (最低 200 日推奨)
    max_workers : int
        並列ダウンロードスレッド数
    retry_delay : float
        リトライ待機秒数 (指数バックオフの基準)
    """

    def __init__(
        self,
        period_days: int = 252,
        max_workers: int = 8,
        retry_delay: float = 1.0,
    ):
        self.period_days = period_days
        self.max_workers = max_workers
        self.retry_delay = retry_delay

        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError(
                "yfinance が見つかりません。pip install yfinance でインストールしてください。"
            )

    def _fetch_single(self, ticker: str, sector: str, max_retries: int = 3) -> Optional[dict]:
        """1銘柄の最新データを取得する (内部メソッド)"""
        end = datetime.today()
        start = end - timedelta(days=self.period_days)

        for attempt in range(max_retries):
            try:
                t = self._yf.Ticker(ticker)
                hist = t.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                )
                if hist is None or hist.empty or len(hist) < 10:
                    return None

                indicators = _compute_indicators(hist)

                # yfinance info (best-effort — may fail/be slow)
                try:
                    info = t.fast_info
                    market_cap = getattr(info, "market_cap", 0) or 0
                    currency = getattr(info, "currency", "USD") or "USD"
                    name = getattr(info, "long_name", ticker) or ticker
                    industry = getattr(info, "industry_disp", sector) or sector
                except Exception:
                    market_cap = 0
                    currency = "JPY" if ticker.endswith(".T") else "USD"
                    name = ticker
                    industry = sector

                return {
                    "Ticker": ticker,
                    "Name": name,
                    "Sector": sector,
                    "Industry": industry,
                    "Market Cap": market_cap,
                    "Currency": currency,
                    **indicators,
                }

            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return None

        return None

    def fetch_sector_snapshot(
        self,
        sector_tickers: dict[str, list[str]],
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        指定されたセクター別銘柄リストからスナップショット DataFrame を作成する。

        Parameters
        ----------
        sector_tickers : dict[str, list[str]]
            セクター名 → ティッカーリスト の辞書
        progress : bool
            進捗表示の有無

        Returns
        -------
        pd.DataFrame
            sector_flow_analyzer.py 互換形式の DataFrame
        """
        tasks: list[tuple[str, str]] = []
        for sector, tickers in sector_tickers.items():
            for ticker in tickers:
                tasks.append((ticker, sector))

        if progress:
            total = len(tasks)
            print(f"  [yfinance] {total} 銘柄を取得中 (workers={self.max_workers})...")

        records = []
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._fetch_single, ticker, sector): (ticker, sector)
                for ticker, sector in tasks
            }
            for future in as_completed(future_to_task):
                ticker, sector = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None:
                        records.append(result)
                except Exception:
                    pass
                completed += 1
                if progress and completed % 20 == 0:
                    print(f"  [yfinance] {completed}/{total} 完了...")

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if progress:
            print(f"  [yfinance] {len(df)} 銘柄のデータ取得完了")
        return df

    def fetch_market_snapshot(
        self,
        market: str = "JP",
        custom_tickers: Optional[dict[str, list[str]]] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        市場別のスナップショット DataFrame を返す。

        Parameters
        ----------
        market : str
            "JP" (日本株) または "US" (米国株)
        custom_tickers : dict, optional
            カスタム銘柄リスト (指定時はデフォルトリストを上書き)
        progress : bool
            進捗表示

        Returns
        -------
        pd.DataFrame
            sector_flow_analyzer.py 互換の DataFrame
        """
        if custom_tickers is not None:
            sector_tickers = custom_tickers
        elif market.upper() == "JP":
            sector_tickers = JP_SECTOR_TICKERS
        elif market.upper() == "US":
            sector_tickers = US_SECTOR_TICKERS
        else:
            raise ValueError(f"Unknown market: {market!r}. Use 'JP' or 'US'.")

        df = self.fetch_sector_snapshot(sector_tickers, progress=progress)

        if df.empty:
            return df

        # マーケット列を設定 (sector_flow_analyzer.py との互換性)
        if "Currency" not in df.columns:
            df["Currency"] = "JPY" if market.upper() == "JP" else "USD"

        return df

    def fetch_tickers(
        self,
        tickers: list[str],
        sector_map: Optional[dict[str, str]] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        任意のティッカーリストからスナップショット DataFrame を返す。

        Parameters
        ----------
        tickers : list[str]
            取得するティッカーリスト
        sector_map : dict[str, str], optional
            ticker → sector の対応表 (未指定時は yfinance の info から取得)
        progress : bool
            進捗表示

        Returns
        -------
        pd.DataFrame
        """
        tasks = [(t, (sector_map or {}).get(t, "Unknown")) for t in tickers]

        if progress:
            print(f"  [yfinance] {len(tasks)} 銘柄を取得中...")

        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._fetch_single, ticker, sector): (ticker, sector)
                for ticker, sector in tasks
            }
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    if result is not None:
                        records.append(result)
                except Exception:
                    pass

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="yfinance データ取得ツール")
    parser.add_argument("--market", choices=["JP", "US"], default="JP",
                        help="取得する市場 (JP/US)")
    parser.add_argument("--out", default=None,
                        help="出力 CSV ファイルパス")
    args = parser.parse_args()

    fetcher = YFinanceFetcher(period_days=252, max_workers=8)
    df = fetcher.fetch_market_snapshot(market=args.market)

    if df.empty:
        print("データの取得に失敗しました。")
    else:
        out_path = args.out or f"screener/snapshots/yfinance_{args.market}_{datetime.today().strftime('%Y-%m-%d')}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n{len(df)} 銘柄のデータを保存: {out_path}")
        print(df[["Ticker", "Sector", "Close", "Change %", "RSI", "ADX"]].head(10).to_string())

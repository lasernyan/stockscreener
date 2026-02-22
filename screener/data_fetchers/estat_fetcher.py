"""
e-STAT API Fetcher
==================
政府統計の総合窓口 (e-STAT) API v3 から景気動向・生産指数などを取得し、
セクター分析にマクロ経済コンテキストを付加する。

主要取得データ:
    1. 景気動向指数 (先行・一致・遅行)        — Cabinet Office
    2. 鉱工業生産指数                          — METI
    3. 消費者物価指数 (CPI)                    — MIC
    4. 有効求人倍率                            — MHLW

API エンドポイント:
    https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData
        ?appId={APP_ID}
        &statsDataId={DATASET_ID}
        &cdTime={TIME_CODE}

API Key 取得:
    https://www.e-stat.go.jp/api/

環境変数:
    ESTAT_APP_ID  — e-STAT アプリケーション ID (必須)

セクター影響マッピング:
    景気動向指数 (先行) 上昇 → 景気敏感セクター (製造業・資本財・金融) 強気
    鉱工業生産指数 上昇     → 素材・鉄鋼・機械 強気
    CPI 上昇              → ディフェンシブ (食品・日用品) 中立、エネルギー 強気
    有効求人倍率 上昇      → 一般消費財・小売 強気

Usage:
    fetcher = EStatFetcher(app_id="YOUR_ID")
    indicators = fetcher.fetch_key_indicators()
    macro_scores = fetcher.get_sector_macro_scores(indicators)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Known e-STAT Dataset IDs
# ---------------------------------------------------------------------------

# 統計データID (e-STAT 公開データセット)
ESTAT_DATASETS = {
    # 景気動向指数 (Cabinet Office - 内閣府)
    "business_cycle_leading":    "0003411084",  # 先行指数
    "business_cycle_coincident": "0003411085",  # 一致指数
    "business_cycle_lagging":    "0003411086",  # 遅行指数

    # 鉱工業生産指数 (METI - 経済産業省)
    "industrial_production":     "0003427513",  # 生産指数

    # 消費者物価指数 (MIC - 総務省)
    "cpi_total":                 "0003427516",  # 総合CPI
    "cpi_ex_food_energy":        "0003427517",  # コアCPI

    # 有効求人倍率 (MHLW - 厚生労働省)
    "job_openings_ratio":        "0004028594",  # 有効求人倍率

    # 実質GDP成長率 (Cabinet Office)
    "gdp_growth":                "0003109737",  # GDP 四半期成長率
}

# セクター感応度マトリクス
# 各指標の変化方向 (+1=上昇, -1=下降) に対するセクター影響度 (-1.0 ~ +1.0)
SECTOR_MACRO_SENSITIVITY: dict[str, dict[str, float]] = {
    "business_cycle_leading": {
        # yfinance / 内部セクター名
        "機械・資本財":      +1.0,
        "鉄鋼・非鉄":        +0.9,
        "素材・化学":         +0.8,
        "電機・精密機器":     +0.8,
        "電機・家電":         +0.7,
        "金融・銀行":         +0.7,
        "自動車・輸送機器":   +0.7,
        "建設・資材":         +0.6,
        "商社・卸売":         +0.6,
        "情報通信":           +0.5,
        "運輸・物流":         +0.5,
        "小売・消費":         +0.4,
        "不動産":             +0.4,
        "食品・生活必需品":   +0.1,
        "ヘルスケア・医薬品": +0.1,
        "公益・電力":         -0.1,
        "エネルギー資源":     +0.5,
        # TradingView JP セクター名
        "生産財製造":         +1.0,
        "素材産業":           +0.9,
        "非エネルギー鉱物":   +0.8,
        "エネルギー鉱物":     +0.5,
        "電子テクノロジー":   +0.8,
        "金融":               +0.7,
        "交通・輸送":         +0.6,
        "流通サービス":       +0.5,
        "耐久消費財":         +0.7,
        "小売業":             +0.4,
        "非耐久消費財":       +0.2,
        "産業サービス":       +0.6,
        "公益事業":           -0.1,
        "ヘルスケアテクノロジー": +0.3,
        "ヘルスケアサービス": +0.1,
        "テクノロジーサービス": +0.5,
        "通信":               +0.4,
        "商業サービス":       +0.4,
        "消費者サービス":     +0.4,
        # US
        "Technology":            +0.7,
        "Financials":            +0.8,
        "Industrials":           +1.0,
        "Materials":             +0.9,
        "Consumer Discretionary": +0.6,
        "Consumer Staples":      +0.1,
        "Healthcare":            +0.1,
        "Utilities":             -0.1,
        "Real Estate":           +0.3,
        "Energy":                +0.5,
        "Communication Services": +0.4,
    },
    "industrial_production": {
        # yfinance / 内部セクター名
        "素材・化学":         +1.0,
        "鉄鋼・非鉄":         +1.0,
        "機械・資本財":       +0.9,
        "電機・精密機器":     +0.8,
        "電機・家電":         +0.7,
        "自動車・輸送機器":   +0.8,
        "建設・資材":         +0.6,
        "エネルギー資源":     +0.7,
        # TradingView JP セクター名
        "素材産業":           +1.0,
        "鉄鋼・非鉄":         +1.0,
        "非エネルギー鉱物":   +0.9,
        "生産財製造":         +0.9,
        "電子テクノロジー":   +0.8,
        "耐久消費財":         +0.8,
        "エネルギー鉱物":     +0.7,
        "交通・輸送":         +0.5,
        "産業サービス":       +0.5,
        # US
        "Industrials":        +1.0,
        "Materials":          +1.0,
        "Energy":             +0.7,
        "Technology":         +0.6,
    },
    "cpi_total": {
        # yfinance / 内部セクター名
        "エネルギー資源":     +0.8,  # インフレ = エネルギー高
        "素材・化学":         +0.7,
        "食品・生活必需品":   +0.6,
        "不動産":             +0.4,
        "公益・電力":         +0.3,
        "金融・銀行":         +0.2,  # 金利上昇 → 利ザヤ拡大
        "小売・消費":         -0.3,  # コスト上昇
        "情報通信":           -0.2,
        # TradingView JP セクター名
        "エネルギー鉱物":     +0.8,
        "素材産業":           +0.7,
        "非エネルギー鉱物":   +0.6,
        "非耐久消費財":       +0.5,
        "公益事業":           +0.3,
        "金融":               +0.2,
        "小売業":             -0.3,
        "テクノロジーサービス": -0.2,
        "消費者サービス":     -0.2,
        # US
        "Energy":             +0.9,
        "Materials":          +0.7,
        "Consumer Staples":   +0.5,
        "Financials":         +0.3,
        "Consumer Discretionary": -0.4,
        "Technology":         -0.3,
    },
    "job_openings_ratio": {
        # yfinance / 内部セクター名
        "小売・消費":         +0.8,
        "自動車・輸送機器":   +0.6,
        "食品・生活必需品":   +0.5,
        "情報通信":           +0.4,
        "運輸・物流":         +0.4,
        "金融・銀行":         +0.3,
        # TradingView JP セクター名
        "小売業":             +0.8,
        "耐久消費財":         +0.6,
        "非耐久消費財":       +0.5,
        "交通・輸送":         +0.4,
        "流通サービス":       +0.4,
        "消費者サービス":     +0.5,
        "金融":               +0.3,
        "テクノロジーサービス": +0.4,
        "商業サービス":       +0.3,
        # US
        "Consumer Discretionary": +0.8,
        "Consumer Staples":   +0.5,
        "Financials":         +0.4,
        "Industrials":        +0.5,
        "Technology":         +0.3,
    },
}

# e-STAT API ベース URL
ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class MacroIndicator:
    """マクロ経済指標の最新値"""
    name: str               # 指標名 (景気動向指数 先行、等)
    dataset_key: str        # 内部キー
    value: float            # 最新値
    prev_value: float       # 前期値
    date: str               # データ日付
    unit: str               # 単位
    change: float           # 前期比変化量
    change_pct: float       # 前期比変化率 (%)
    trend: str              # "上昇" / "横ばい" / "下降"
    source: str             # 公表機関

    @property
    def direction(self) -> int:
        """変化方向: +1=上昇, 0=横ばい, -1=下降"""
        if self.change_pct > 0.1:
            return 1
        elif self.change_pct < -0.1:
            return -1
        return 0


# ---------------------------------------------------------------------------
# Main Fetcher Class
# ---------------------------------------------------------------------------

class EStatFetcher:
    """
    e-STAT API v3 からマクロ経済統計を取得するクラス。

    Parameters
    ----------
    app_id : str, optional
        e-STAT アプリケーション ID。
        未指定時は環境変数 ESTAT_APP_ID を参照。
    timeout : int
        HTTP リクエストタイムアウト (秒)
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        timeout: int = 30,
    ):
        self.app_id = app_id or os.environ.get("ESTAT_APP_ID", "")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "SectorRotationAnalyzer/1.0",
        })

    def _get(self, params: dict) -> Optional[dict]:
        """e-STAT API への GET リクエスト"""
        if not self.app_id:
            return None

        params["appId"] = self.app_id
        url = f"{ESTAT_BASE_URL}/getStatsData"

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # e-STAT API はエラーレスポンスも 200 で返す場合がある
            status_code = data.get("GET_STATS_DATA", {}).get("PARAMETER", {}).get("RESULT", {}).get("STATUS")
            if status_code and int(status_code) != 0:
                err_msg = data.get("GET_STATS_DATA", {}).get("PARAMETER", {}).get("RESULT", {}).get("ERROR_MSG", "")
                print(f"  [e-STAT] API エラー (status={status_code}): {err_msg}")
                return None

            return data
        except requests.exceptions.HTTPError as e:
            print(f"  [e-STAT] HTTP エラー: {e}")
            return None
        except requests.exceptions.ConnectionError:
            print("  [e-STAT] 接続エラー: ネットワークを確認してください")
            return None
        except Exception as e:
            print(f"  [e-STAT] 予期しないエラー: {e}")
            return None

    def _parse_stats_data(
        self,
        data: dict,
        dataset_key: str,
        name: str,
        source: str,
        unit: str = "index",
    ) -> Optional[MacroIndicator]:
        """e-STAT API レスポンスをパースして MacroIndicator に変換する"""
        try:
            stats = data.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
            data_inf = stats.get("DATA_INF", {})
            value_list = data_inf.get("VALUE", [])

            if not value_list:
                return None

            # 最新2件を取得
            if isinstance(value_list, list):
                # 時間コード順に並んでいることを前提
                values_with_time = []
                for v in value_list:
                    if isinstance(v, dict):
                        val_str = v.get("$", "").replace(",", "")
                        time_code = v.get("@time", v.get("@TIME", ""))
                        try:
                            val = float(val_str)
                            values_with_time.append((time_code, val))
                        except (ValueError, TypeError):
                            pass

                if len(values_with_time) < 2:
                    return None

                # 最新2件
                values_with_time.sort(key=lambda x: x[0], reverse=True)
                latest_time, latest_val = values_with_time[0]
                prev_time, prev_val = values_with_time[1]

            else:
                return None

            change = latest_val - prev_val
            change_pct = (change / prev_val * 100.0) if prev_val != 0 else 0.0

            if change_pct > 0.1:
                trend = "上昇"
            elif change_pct < -0.1:
                trend = "下降"
            else:
                trend = "横ばい"

            # 日付の整形 (時間コード: "2025100000" → "2025-10")
            date_str = latest_time
            if len(date_str) >= 6:
                year = date_str[:4]
                month = date_str[4:6]
                date_str = f"{year}-{month}"

            return MacroIndicator(
                name=name,
                dataset_key=dataset_key,
                value=round(latest_val, 2),
                prev_value=round(prev_val, 2),
                date=date_str,
                unit=unit,
                change=round(change, 2),
                change_pct=round(change_pct, 2),
                trend=trend,
                source=source,
            )
        except Exception as e:
            print(f"  [e-STAT] パースエラー ({dataset_key}): {e}")
            return None

    def fetch_business_cycle_index(self) -> list[MacroIndicator]:
        """景気動向指数 (先行・一致・遅行) を取得する"""
        indicators = []
        configs = [
            ("business_cycle_leading",    "景気動向指数 (先行)",  "内閣府"),
            ("business_cycle_coincident", "景気動向指数 (一致)",  "内閣府"),
            ("business_cycle_lagging",    "景気動向指数 (遅行)",  "内閣府"),
        ]

        for key, name, source in configs:
            data = self._get({
                "statsDataId": ESTAT_DATASETS[key],
                "metaGetFlg": "N",
                "cnt": 3,
                "startPosition": 1,
            })
            if data:
                ind = self._parse_stats_data(data, key, name, source, unit="DI")
                if ind:
                    indicators.append(ind)

        return indicators

    def fetch_industrial_production(self) -> Optional[MacroIndicator]:
        """鉱工業生産指数を取得する"""
        data = self._get({
            "statsDataId": ESTAT_DATASETS["industrial_production"],
            "metaGetFlg": "N",
            "cnt": 3,
        })
        if data:
            return self._parse_stats_data(
                data, "industrial_production",
                "鉱工業生産指数", "経済産業省", unit="2015=100"
            )
        return None

    def fetch_cpi(self) -> Optional[MacroIndicator]:
        """消費者物価指数 (総合) を取得する"""
        data = self._get({
            "statsDataId": ESTAT_DATASETS["cpi_total"],
            "metaGetFlg": "N",
            "cnt": 3,
        })
        if data:
            return self._parse_stats_data(
                data, "cpi_total",
                "消費者物価指数 (総合)", "総務省", unit="2020=100"
            )
        return None

    def fetch_job_openings_ratio(self) -> Optional[MacroIndicator]:
        """有効求人倍率を取得する"""
        data = self._get({
            "statsDataId": ESTAT_DATASETS["job_openings_ratio"],
            "metaGetFlg": "N",
            "cnt": 3,
        })
        if data:
            return self._parse_stats_data(
                data, "job_openings_ratio",
                "有効求人倍率", "厚生労働省", unit="倍"
            )
        return None

    def fetch_key_indicators(self, verbose: bool = True) -> list[MacroIndicator]:
        """
        主要マクロ指標を一括取得する。

        Returns
        -------
        list[MacroIndicator]
            取得できた指標のリスト (API キーなし / エラー時は空リスト)
        """
        if not self.app_id:
            if verbose:
                print("  [e-STAT] ESTAT_APP_ID が設定されていません。マクロデータをスキップします。")
                print("           APIキー取得: https://www.e-stat.go.jp/api/")
            return []

        if verbose:
            print("  [e-STAT] マクロ経済指標を取得中...")

        indicators: list[MacroIndicator] = []

        # 景気動向指数
        bci = self.fetch_business_cycle_index()
        indicators.extend(bci)

        # 鉱工業生産指数
        ipi = self.fetch_industrial_production()
        if ipi:
            indicators.append(ipi)

        # CPI
        cpi = self.fetch_cpi()
        if cpi:
            indicators.append(cpi)

        # 有効求人倍率
        jor = self.fetch_job_openings_ratio()
        if jor:
            indicators.append(jor)

        if verbose:
            print(f"  [e-STAT] {len(indicators)} 件の指標を取得")

        return indicators

    def get_sector_macro_scores(
        self,
        indicators: list[MacroIndicator],
    ) -> dict[str, float]:
        """
        マクロ指標からセクター別マクロ影響スコアを計算する。

        各指標の変化方向 × セクター感応度 を合算し、
        -10 ~ +10 の範囲でセクター強弱を調整するスコアを返す。

        Parameters
        ----------
        indicators : list[MacroIndicator]

        Returns
        -------
        dict[str, float]
            sector → macro_score (-10 ~ +10)
        """
        sector_scores: dict[str, float] = {}

        for indicator in indicators:
            sensitivities = SECTOR_MACRO_SENSITIVITY.get(indicator.dataset_key, {})
            direction = indicator.direction  # +1, 0, -1
            magnitude = min(abs(indicator.change_pct) / 2.0, 1.0)  # 正規化

            for sector, sensitivity in sensitivities.items():
                contribution = direction * sensitivity * magnitude * 10.0
                sector_scores[sector] = sector_scores.get(sector, 0.0) + contribution

        # クランプ -10 ~ +10
        return {s: max(-10.0, min(10.0, v)) for s, v in sector_scores.items()}

    def print_summary(self, indicators: list[MacroIndicator]) -> None:
        """マクロ指標サマリーをターミナルに出力する"""
        if not indicators:
            print("  [e-STAT] マクロデータなし")
            return

        print(f"\n  マクロ経済指標サマリー ({datetime.today().strftime('%Y-%m-%d')})")
        print(f"  {'─' * 70}")
        print(f"  {'指標':<28} {'最新値':>8} {'前期値':>8} {'変化率':>8}  トレンド")
        print(f"  {'─' * 70}")

        for ind in indicators:
            trend_icon = "▲" if ind.trend == "上昇" else ("▼" if ind.trend == "下降" else "─")
            print(
                f"  {ind.name:<28} {ind.value:>8.2f} {ind.prev_value:>8.2f} "
                f"{ind.change_pct:>+7.2f}%  {trend_icon} {ind.trend}"
            )

        print(f"  {'─' * 70}")


# ---------------------------------------------------------------------------
# Fallback: Simulated indicators for offline/no-key mode
# ---------------------------------------------------------------------------

def get_simulated_indicators() -> list[MacroIndicator]:
    """
    APIキーなしでのテスト用: ダミーのマクロ指標を返す。
    実際の分析では使用しないこと。
    """
    today = datetime.today().strftime("%Y-%m")
    return [
        MacroIndicator(
            name="景気動向指数 (先行) [シミュレーション]",
            dataset_key="business_cycle_leading",
            value=100.5, prev_value=99.8, date=today,
            unit="DI", change=0.7, change_pct=0.70,
            trend="上昇", source="シミュレーション",
        ),
        MacroIndicator(
            name="鉱工業生産指数 [シミュレーション]",
            dataset_key="industrial_production",
            value=102.3, prev_value=101.8, date=today,
            unit="2015=100", change=0.5, change_pct=0.49,
            trend="上昇", source="シミュレーション",
        ),
        MacroIndicator(
            name="消費者物価指数 [シミュレーション]",
            dataset_key="cpi_total",
            value=107.2, prev_value=106.9, date=today,
            unit="2020=100", change=0.3, change_pct=0.28,
            trend="上昇", source="シミュレーション",
        ),
    ]


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="e-STAT マクロ経済データ取得ツール")
    parser.add_argument("--app-id", default=None,
                        help="e-STAT アプリケーション ID (環境変数 ESTAT_APP_ID でも可)")
    parser.add_argument("--simulate", action="store_true",
                        help="シミュレーションデータを使用 (APIキー不要)")
    args = parser.parse_args()

    if args.simulate:
        indicators = get_simulated_indicators()
    else:
        app_id = args.app_id or os.environ.get("ESTAT_APP_ID", "")
        if not app_id:
            print("警告: ESTAT_APP_ID が設定されていません。")
            print("  --simulate オプションでシミュレーションデータを使用できます。")

        fetcher = EStatFetcher(app_id=app_id)
        indicators = fetcher.fetch_key_indicators()

    if indicators:
        fetcher_for_print = EStatFetcher()
        fetcher_for_print.print_summary(indicators)

        # セクタースコアの例
        scores = EStatFetcher().get_sector_macro_scores(indicators)
        print("\n  セクター別マクロスコア:")
        for sector, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "+" * int(max(0, score)) + "-" * int(max(0, -score))
            print(f"    {sector:<30} {score:+.1f}  {bar}")

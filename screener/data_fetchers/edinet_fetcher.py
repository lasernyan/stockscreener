"""
EDINET API Fetcher
==================
金融庁 EDINET API v2 から企業イベント情報を取得し、
セクター分析へのイベントスタディ・シグナルを生成する。

対応イベント:
    - 公開買付届出書 (TOB)   → 対象企業・セクターへの強気シグナル
    - 大量保有報告書          → 機関投資家の大量取得 → 中程度の強気シグナル
    - 自社株取得状況報告書    → 自社株買い → 経営陣の自信の表れ (強気)
    - 変更報告書(大量保有)    → 保有比率の変化を追跡

EDINET API v2 エンドポイント:
    GET https://api.edinet-api.go.jp/api/v2/documents.json
        ?date=YYYY-MM-DD
        &type=2
        &Subscription-Key={API_KEY}

API Key 取得:
    https://disclosure2dl.edinet-api.go.jp/guide/static/disclosure/REVISION_01/index.html

環境変数:
    EDINET_API_KEY  — EDINET API サブスクリプションキー (必須)

Usage:
    fetcher = EdinetFetcher(api_key="YOUR_KEY")
    events = fetcher.fetch_recent_events(days=7)
    for event in events:
        print(event)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

# EDINET 書類種別コード (docTypeCode)
EDINET_DOC_TYPES = {
    # 公開買付
    "010": "公開買付届出書",
    "011": "公開買付撤回届出書",
    "012": "公開買付報告書",
    "013": "意見表明報告書",
    "014": "対質問回答報告書",
    "015": "訂正公開買付届出書",
    "016": "訂正公開買付報告書",
    # 大量保有報告書
    "030": "大量保有報告書",
    "031": "大量保有報告書(特例報告書)",
    "032": "大量保有報告書(変更報告書)",
    "033": "大量保有報告書(特例変更報告書)",
    # 自己株式
    "070": "自己株式取得状況報告書",
    "071": "自己株式の取得等の状況報告書",
}

# イベント分類 (シグナル生成用)
EVENT_CATEGORIES = {
    "010": "TOB",           # 公開買付届出書
    "011": "TOB_CANCEL",    # 公開買付撤回
    "012": "TOB_REPORT",    # 公開買付報告書
    "013": "TOB_OPINION",   # 意見表明
    "030": "LARGE_HOLDING", # 大量保有
    "031": "LARGE_HOLDING",
    "032": "LARGE_HOLDING_CHANGE",
    "033": "LARGE_HOLDING_CHANGE",
    "070": "BUYBACK",       # 自社株買い
    "071": "BUYBACK",
}

# シグナル強度マッピング (0-100)
SIGNAL_STRENGTH = {
    "TOB":                   90,  # 最強: 公開買付は即座のプレミアム
    "BUYBACK":               65,  # 強い: 経営陣が割安と判断
    "LARGE_HOLDING":         50,  # 中程度: 機関投資家の新規取得
    "LARGE_HOLDING_CHANGE":  30,  # 弱い: 保有比率の変化
    "TOB_REPORT":            25,  # 情報のみ
    "TOB_CANCEL":           -50,  # ネガティブ: 買付撤回
    "TOB_OPINION":           20,
}

# EDINET API ベース URL
EDINET_BASE_URL = "https://api.edinet-api.go.jp/api/v2"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class EdinetEvent:
    """EDINET から取得したイベント情報"""
    doc_id: str
    doc_type_code: str
    doc_type_name: str
    category: str               # TOB / LARGE_HOLDING / BUYBACK など
    submit_date: str            # 提出日 YYYY-MM-DD
    filer_name: str             # 提出者名
    filer_edinet_code: str      # 提出者 EDINET コード
    issuer_name: str            # 発行者名 (大量保有・TOB の場合)
    issuer_edinet_code: str     # 発行者 EDINET コード
    ticker: str                 # 推定ティッカー (マッピング後)
    description: str            # 書類種別の説明
    signal_strength: int        # シグナル強度 (0-100、ネガティブも可)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def is_bullish(self) -> bool:
        return self.signal_strength > 0

    @property
    def signal_label(self) -> str:
        if self.signal_strength >= 70:
            return "STRONG_LONG"
        elif self.signal_strength >= 40:
            return "LONG"
        elif self.signal_strength >= 10:
            return "MILD_LONG"
        elif self.signal_strength <= -40:
            return "STRONG_SHORT"
        elif self.signal_strength < 0:
            return "SHORT"
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# EDINET Code → Ticker マッピング (主要銘柄)
# ---------------------------------------------------------------------------

# EDINET コード → 東証ティッカーの対応表 (一部抜粋)
# フル版は EDINET の会社情報 API から取得可能
EDINET_TO_TICKER: dict[str, str] = {
    "E02166": "7203.T",  # トヨタ自動車
    "E04737": "7267.T",  # 本田技研工業
    "E00436": "7201.T",  # 日産自動車
    "E02765": "6758.T",  # ソニーグループ
    "E01496": "8306.T",  # 三菱UFJフィナンシャル・グループ
    "E03527": "8316.T",  # 三井住友フィナンシャルグループ
    "E03746": "8411.T",  # みずほフィナンシャルグループ
    "E02212": "9432.T",  # NTT
    "E04428": "9984.T",  # ソフトバンクグループ
    "E01488": "4502.T",  # 武田薬品工業
    "E03814": "4568.T",  # 第一三共
    "E02726": "4503.T",  # アステラス製薬
    "E02928": "4519.T",  # 中外製薬
    "E00950": "5401.T",  # 日本製鉄
    "E01244": "5411.T",  # JFEホールディングス
    "E04019": "8801.T",  # 三井不動産
    "E00028": "8802.T",  # 三菱地所
    "E02331": "8058.T",  # 三菱商事
    "E02607": "8031.T",  # 三井物産
    "E04818": "8001.T",  # 伊藤忠商事
    "E04568": "9983.T",  # ファーストリテイリング
    "E03529": "3382.T",  # セブン&アイ・ホールディングス
    "E00386": "5020.T",  # ENEOSホールディングス
    "E02841": "1605.T",  # INPEX
    "E01264": "4063.T",  # 信越化学工業
}


# ---------------------------------------------------------------------------
# Main Fetcher Class
# ---------------------------------------------------------------------------

class EdinetFetcher:
    """
    EDINET API v2 からイベント情報を取得するクラス。

    Parameters
    ----------
    api_key : str, optional
        EDINET API サブスクリプションキー。
        未指定時は環境変数 EDINET_API_KEY を参照。
    timeout : int
        HTTP リクエストタイムアウト (秒)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get("EDINET_API_KEY", "")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "SectorRotationAnalyzer/1.0",
        })

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """EDINET API への GET リクエスト"""
        if self.api_key:
            params["Subscription-Key"] = self.api_key

        url = f"{EDINET_BASE_URL}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                print(f"  [EDINET] 認証エラー: APIキーを確認してください (EDINET_API_KEY)")
            elif e.response is not None and e.response.status_code == 404:
                print(f"  [EDINET] データなし: {params.get('date', '?')}")
            else:
                print(f"  [EDINET] HTTP エラー: {e}")
            return None
        except requests.exceptions.ConnectionError:
            print("  [EDINET] 接続エラー: ネットワークを確認してください")
            return None
        except Exception as e:
            print(f"  [EDINET] 予期しないエラー: {e}")
            return None

    def _parse_document(self, doc: dict) -> Optional[EdinetEvent]:
        """API レスポンスの1件をパースして EdinetEvent に変換する"""
        doc_type_code = str(doc.get("docTypeCode", ""))

        if doc_type_code not in EDINET_DOC_TYPES:
            return None

        category = EVENT_CATEGORIES.get(doc_type_code, "OTHER")
        strength = SIGNAL_STRENGTH.get(category, 0)

        # 発行者情報 (TOB・大量保有の場合は issuer が対象企業)
        issuer_name = doc.get("issuerName") or doc.get("filerName", "")
        issuer_edinet = doc.get("issuerEdinetCode") or doc.get("edinetCode", "")

        # ティッカーの推定
        ticker = EDINET_TO_TICKER.get(issuer_edinet, "")
        if not ticker and doc.get("issuerEdinetCode"):
            ticker = self._lookup_ticker(doc["issuerEdinetCode"])

        # 提出日の抽出 (submitDateTime: "YYYY-MM-DD HH:MM:SS")
        submit_dt = doc.get("submitDateTime", "")
        submit_date = submit_dt[:10] if submit_dt else doc.get("periodEnd", "")

        return EdinetEvent(
            doc_id=doc.get("docID", ""),
            doc_type_code=doc_type_code,
            doc_type_name=EDINET_DOC_TYPES.get(doc_type_code, doc_type_code),
            category=category,
            submit_date=submit_date,
            filer_name=doc.get("filerName", ""),
            filer_edinet_code=doc.get("edinetCode", ""),
            issuer_name=issuer_name,
            issuer_edinet_code=issuer_edinet,
            ticker=ticker,
            description=doc.get("docDescription") or EDINET_DOC_TYPES.get(doc_type_code, ""),
            signal_strength=strength,
            raw=doc,
        )

    def _lookup_ticker(self, edinet_code: str) -> str:
        """
        EDINET コードからティッカーを検索する (ベストエフォート)。
        内部マッピングで見つからない場合は空文字を返す。
        """
        return EDINET_TO_TICKER.get(edinet_code, "")

    def fetch_daily_events(
        self,
        date: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[EdinetEvent]:
        """
        指定日の EDINET 提出書類を取得する。

        Parameters
        ----------
        date : str, optional
            対象日 "YYYY-MM-DD"。未指定時は今日。
        doc_types : list[str], optional
            取得する書類種別コードのリスト。
            未指定時はデフォルト対象 (TOB・大量保有・自社株買い) を全て取得。

        Returns
        -------
        list[EdinetEvent]
        """
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        target_types = set(doc_types or list(EVENT_CATEGORIES.keys()))

        data = self._get("documents.json", {"date": date, "type": "2"})
        if data is None:
            return []

        results = data.get("results", [])
        if not results:
            return []

        events = []
        for doc in results:
            doc_type = str(doc.get("docTypeCode", ""))
            if doc_type not in target_types:
                continue
            event = self._parse_document(doc)
            if event is not None:
                events.append(event)

        return events

    def fetch_recent_events(
        self,
        days: int = 7,
        doc_types: Optional[list[str]] = None,
        skip_weekends: bool = True,
    ) -> list[EdinetEvent]:
        """
        直近 N 日分のイベントをまとめて取得する。

        Parameters
        ----------
        days : int
            遡る営業日数
        doc_types : list[str], optional
            取得対象の書類種別コード
        skip_weekends : bool
            土日をスキップするか

        Returns
        -------
        list[EdinetEvent]
            日付の新しい順にソートされたイベントリスト
        """
        all_events: list[EdinetEvent] = []
        today = datetime.today()
        checked = 0

        for delta in range(days * 2):  # 余裕を持って日付をイテレート
            if checked >= days:
                break
            target = today - timedelta(days=delta)
            if skip_weekends and target.weekday() >= 5:  # 土=5, 日=6
                continue

            checked += 1
            date_str = target.strftime("%Y-%m-%d")
            events = self.fetch_daily_events(date=date_str, doc_types=doc_types)
            all_events.extend(events)

            if delta < days * 2 - 1:
                time.sleep(0.5)  # API レートリミット対策

        # 日付の新しい順にソート
        all_events.sort(key=lambda e: e.submit_date, reverse=True)
        return all_events

    def get_sector_event_boosts(
        self,
        events: list[EdinetEvent],
        sector_ticker_map: dict[str, str],
    ) -> dict[str, float]:
        """
        イベントリストからセクター別のイベントブーストスコアを計算する。

        Parameters
        ----------
        events : list[EdinetEvent]
        sector_ticker_map : dict[str, str]
            ticker → sector の対応表

        Returns
        -------
        dict[str, float]
            sector → boost_score (正=強気ブースト, 負=弱気ブースト)
        """
        sector_boosts: dict[str, float] = {}

        for event in events:
            if not event.ticker or event.ticker not in sector_ticker_map:
                continue
            sector = sector_ticker_map[event.ticker]
            boost = event.signal_strength * 0.1  # 0-9 点のブースト
            sector_boosts[sector] = sector_boosts.get(sector, 0.0) + boost

        return sector_boosts

    def get_ticker_event_signals(
        self,
        events: list[EdinetEvent],
    ) -> dict[str, list[EdinetEvent]]:
        """
        イベントをティッカー別に整理する。

        Returns
        -------
        dict[str, list[EdinetEvent]]
            ticker → events のマッピング
        """
        ticker_events: dict[str, list[EdinetEvent]] = {}
        for event in events:
            if event.ticker:
                ticker_events.setdefault(event.ticker, []).append(event)
        return ticker_events

    def print_summary(self, events: list[EdinetEvent]) -> None:
        """イベントサマリーをターミナルに出力する"""
        if not events:
            print("  [EDINET] イベントなし")
            return

        print(f"\n  EDINET イベントサマリー ({len(events)} 件)")
        print(f"  {'─' * 70}")

        tob_events = [e for e in events if e.category == "TOB"]
        buyback_events = [e for e in events if e.category == "BUYBACK"]
        holding_events = [e for e in events if "HOLDING" in e.category]

        if tob_events:
            print(f"\n  ▶ 公開買付 (TOB) — {len(tob_events)} 件")
            for e in tob_events[:5]:
                sign = "+" if e.is_bullish else "-"
                print(f"    [{e.submit_date}] {e.issuer_name} "
                      f"({e.ticker or 'N/A'}) {sign}{abs(e.signal_strength):.0f}pt")

        if buyback_events:
            print(f"\n  ▶ 自社株買い — {len(buyback_events)} 件")
            for e in buyback_events[:5]:
                print(f"    [{e.submit_date}] {e.issuer_name} "
                      f"({e.ticker or 'N/A'}) +{e.signal_strength:.0f}pt")

        if holding_events:
            print(f"\n  ▶ 大量保有報告書 — {len(holding_events)} 件")
            for e in holding_events[:5]:
                print(f"    [{e.submit_date}] {e.issuer_name} "
                      f"({e.ticker or 'N/A'}) +{e.signal_strength:.0f}pt "
                      f"[{e.doc_type_name}]")

        print(f"  {'─' * 70}")


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EDINET イベント取得ツール")
    parser.add_argument("--days", type=int, default=5,
                        help="遡る日数 (デフォルト: 5)")
    parser.add_argument("--date", default=None,
                        help="特定の日付 YYYY-MM-DD")
    parser.add_argument("--api-key", default=None,
                        help="EDINET API キー (環境変数 EDINET_API_KEY でも可)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("EDINET_API_KEY", "")
    if not api_key:
        print("警告: EDINET_API_KEY が設定されていません。")
        print("  export EDINET_API_KEY=your_key または --api-key オプションで指定してください。")
        print("  APIキーなしでもテスト実行します (認証エラーになる可能性あり)...")

    fetcher = EdinetFetcher(api_key=api_key)

    if args.date:
        print(f"  {args.date} のイベントを取得中...")
        events = fetcher.fetch_daily_events(date=args.date)
    else:
        print(f"  直近 {args.days} 日間のイベントを取得中...")
        events = fetcher.fetch_recent_events(days=args.days)

    fetcher.print_summary(events)

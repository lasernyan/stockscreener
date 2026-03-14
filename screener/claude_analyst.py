#!/usr/bin/env python3
"""
Claude AI Market Analyst
=========================
Anthropic Claude API を使って、セクターメトリクス・ローテーションシグナル・
トレード候補を分析し、日本語で詳細な市場分析レポートを生成する。

Uses:
  - Model    : claude-opus-4-6
  - Thinking : adaptive (深い推論を自動的に活用)
  - Streaming: 長い出力のタイムアウトを防止

Requirements:
  pip install anthropic

Environment:
  ANTHROPIC_API_KEY — Anthropic APIキー (必須)

Usage:
  python3 screener/claude_analyst.py                    # 最新データで分析
  python3 screener/claude_analyst.py --market JP        # 日本市場のみ
  python3 screener/claude_analyst.py --market ALL       # JP + US
  python3 screener/claude_analyst.py --no-stream        # ストリーミング無効
  python3 screener/claude_analyst.py --export-html      # HTMLレポート生成
  python3 screener/claude_analyst.py --backtest         # バックテスト結果も分析
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

try:
    import anthropic
except ImportError:
    print("[ERROR] anthropic パッケージが必要です: pip install anthropic")
    sys.exit(1)

import pandas as pd

# ============================================================
# データ読み込みユーティリティ
# ============================================================

def _load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def load_market_data(screener_dir: str, market: str) -> dict:
    """
    screener/ ディレクトリから最新の分析結果を読み込む。
    市場は "JP", "US", "ALL" のいずれか。
    """
    markets = ["JP", "US"] if market.upper() == "ALL" else [market.upper()]
    data = {}

    for mkt in markets:
        suffix = f"_{mkt.lower()}"
        d = {
            "market": mkt,
            "sector_metrics": _load_csv_safe(os.path.join(screener_dir, f"sector_metrics{suffix}.csv")),
            "trade_candidates": _load_csv_safe(os.path.join(screener_dir, f"trade_candidates{suffix}.csv")),
        }
        # sector_metrics (汎用)
        if d["sector_metrics"] is None:
            d["sector_metrics"] = _load_csv_safe(os.path.join(screener_dir, "sector_metrics.csv"))
        if d["trade_candidates"] is None:
            d["trade_candidates"] = _load_csv_safe(os.path.join(screener_dir, "trade_candidates.csv"))

        data[mkt] = d

    return data


def load_backtest_results(screener_dir: str, market: str) -> Optional[dict]:
    """バックテスト結果の CSV を読み込む"""
    path = os.path.join(screener_dir, f"backtest_trades_{market}.csv")
    df = _load_csv_safe(path)
    if df is None or df.empty:
        return None

    total_trades = len(df)
    wins = (df["pnl_pct"] > 0).sum() if "pnl_pct" in df.columns else 0
    total_pnl = df["pnl_pct"].sum() if "pnl_pct" in df.columns else 0

    return {
        "total_trades": total_trades,
        "win_rate": round(wins / total_trades * 100, 1) if total_trades else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / total_trades, 4) if total_trades else 0,
        "best_sector": df.groupby("sector")["pnl_pct"].sum().idxmax() if "sector" in df.columns else "N/A",
    }


def _df_to_compact_str(df: pd.DataFrame, max_rows: int = 20) -> str:
    """DataFrame を Claude に渡すためのコンパクトな文字列に変換"""
    if df is None or df.empty:
        return "(データなし)"
    # 上位 max_rows 行のみ
    df = df.head(max_rows)
    # NaN を空文字に
    df = df.fillna("")
    return df.to_csv(index=False)


# ============================================================
# プロンプト構築
# ============================================================

SYSTEM_PROMPT = """あなたは機関投資家レベルの市場アナリストです。
セクターローテーション、資金フロー分析、マルチアセット投資戦略の専門家として、
提供されたデータを深く分析し、実践的かつ具体的な投資インサイトを日本語で提供してください。

分析の際は以下の視点を必ず含めてください：
1. マクロ的なリスクオン/リスクオフの状態判断
2. 現在の資金フロー方向と強度
3. セクターローテーションのフェーズ（景気サイクルとの整合性）
4. 暗号資産・コモディティ・FX があればそれらとの相互関係
5. 具体的なアクションプラン（エントリー条件、リスク管理含む）

回答は以下の形式で構造化してください：
- 市場全体の状況サマリー（3-5文）
- セクター分析（強/弱/注目セクター）
- 資金フロー分析
- ロングシグナル（上位3候補、根拠付き）
- ショートシグナル（上位3候補、根拠付き）
- リスクシナリオ（2-3個）
- 総合アクションプラン
"""


def build_analysis_prompt(market_data: dict, backtest: Optional[dict] = None) -> str:
    """分析プロンプトを構築"""
    sections = []
    today = datetime.now().strftime("%Y-%m-%d %H:%M JST")

    sections.append(f"# マーケットデータ分析リクエスト\n日時: {today}\n")

    for mkt, d in market_data.items():
        sections.append(f"\n## {mkt}市場")

        if d["sector_metrics"] is not None and not d["sector_metrics"].empty:
            # 最新日付のみ
            sm = d["sector_metrics"]
            if "date" in sm.columns:
                latest_date = sm["date"].max()
                sm = sm[sm["date"] == latest_date]

            sections.append(f"\n### セクターメトリクス ({mkt})")
            sections.append(_df_to_compact_str(sm, max_rows=15))

        if d["trade_candidates"] is not None and not d["trade_candidates"].empty:
            tc = d["trade_candidates"]
            # LONGとSHORTを分けて上位5件ずつ
            if "direction" in tc.columns:
                long_c  = tc[tc["direction"].str.upper() == "LONG"].head(5)
                short_c = tc[tc["direction"].str.upper() == "SHORT"].head(5)
                sections.append(f"\n### LONG候補 ({mkt})")
                sections.append(_df_to_compact_str(long_c))
                sections.append(f"\n### SHORT候補 ({mkt})")
                sections.append(_df_to_compact_str(short_c))
            else:
                sections.append(f"\n### トレード候補 ({mkt})")
                sections.append(_df_to_compact_str(tc, max_rows=10))

    if backtest:
        sections.append(f"\n## バックテスト結果サマリー")
        sections.append(f"- 総トレード数: {backtest['total_trades']}")
        sections.append(f"- 勝率: {backtest['win_rate']}%")
        sections.append(f"- 累計PnL: {backtest['total_pnl']:+.2f}%")
        sections.append(f"- 平均PnL/trade: {backtest['avg_pnl']:+.4f}%")
        sections.append(f"- ベストセクター: {backtest['best_sector']}")

    sections.append("""
---

上記のデータを基に以下を分析してください：

1. **市場全体サマリー** — 現在のリスクオン/オフ状態、主要テーマ
2. **セクター分析** — SCSスコアの意味、注目すべき変化、ローテーションのフェーズ
3. **資金フロー分析** — どこに資金が流入/流出しているか、その持続性
4. **ロング戦略** — 最も確信度の高い3候補（エントリー条件・利確・損切りレベル含む）
5. **ショート戦略** — 最も確信度の高い3候補（同様に条件付き）
6. **リスクシナリオ** — 現在の分析が崩れる条件、注意すべきイベント
7. **総合アクションプラン** — 今週の優先アクション（具体的かつ実行可能なもの）
""")

    return "\n".join(sections)


# ============================================================
# Claude API 呼び出し
# ============================================================

class ClaudeMarketAnalyst:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-6",
        use_streaming: bool = True,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY が設定されていません。\n"
                "  export ANTHROPIC_API_KEY='your-key'\n"
                "  または --api-key オプションを使用してください。"
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.use_streaming = use_streaming

    def analyze(
        self,
        prompt: str,
        verbose: bool = True,
    ) -> str:
        """
        市場データを分析して詳細なレポートを返す。
        Streaming + adaptive thinking を使用。
        """
        if verbose:
            print(f"\n[Claude {self.model}] 分析開始...\n")
            print("-" * 60)

        if self.use_streaming:
            return self._analyze_streaming(prompt, verbose)
        else:
            return self._analyze_sync(prompt, verbose)

    def _analyze_streaming(self, prompt: str, verbose: bool) -> str:
        """ストリーミング + adaptive thinking"""
        full_text = ""
        thinking_text = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking" and verbose:
                        print("[思考中...]", end="", flush=True)
                    elif event.content_block.type == "text":
                        if thinking_text and verbose:
                            print()  # 思考ブロックの後に改行
                        thinking_text = ""

                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        thinking_text += event.delta.thinking
                        if verbose:
                            print(".", end="", flush=True)
                    elif event.delta.type == "text_delta":
                        full_text += event.delta.text
                        if verbose:
                            print(event.delta.text, end="", flush=True)

            # 最終メッセージからトークン使用量を取得
            final = stream.get_final_message()

        if verbose:
            usage = final.usage
            print(f"\n\n[トークン使用量: 入力={usage.input_tokens:,} / 出力={usage.output_tokens:,}]")

        return full_text

    def _analyze_sync(self, prompt: str, verbose: bool) -> str:
        """非ストリーミング版"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        if verbose:
            print(text)
            usage = response.usage
            print(f"\n[トークン使用量: 入力={usage.input_tokens:,} / 出力={usage.output_tokens:,}]")

        return text

    def quick_signal_check(self, sector_summary: str) -> str:
        """
        セクターサマリーからクイックシグナル判定 (低コスト: Haiku使用)
        戦略上の「今すぐ動くか待つか」の判断に使う。
        """
        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"以下のセクターデータを見て、今すぐロングエントリーすべきか、"
                    f"待つべきかを3行で判定してください（日本語で）:\n\n{sector_summary}"
                )
            }],
        )
        return next((b.text for b in response.content if b.type == "text"), "")


# ============================================================
# HTML レポート生成
# ============================================================

def export_html_analysis(analysis_text: str, output_path: str, market: str):
    """Claude の分析をHTMLファイルに保存"""
    import re

    # Markdown → シンプルなHTMLに変換
    html_body = analysis_text
    html_body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^# (.+)$",  r"<h1>\1</h1>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
    html_body = re.sub(r"\*(.+?)\*",   r"<em>\1</em>", html_body)
    html_body = re.sub(r"^- (.+)$",    r"<li>\1</li>", html_body, flags=re.MULTILINE)
    html_body = html_body.replace("\n\n", "</p><p>")

    now = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Market Analysis — {market} — {now}</title>
  <style>
    body {{
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 30px 20px;
      line-height: 1.7;
    }}
    h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
    h2 {{ color: #79c0ff; margin-top: 30px; border-left: 4px solid #58a6ff; padding-left: 12px; }}
    h3 {{ color: #a8d8ff; }}
    strong {{ color: #ffa657; }}
    em {{ color: #7ee787; font-style: normal; }}
    li {{ margin: 4px 0; }}
    p {{ margin: 10px 0; }}
    .header {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 15px 20px;
      margin-bottom: 25px;
    }}
    .badge {{
      display: inline-block;
      background: #1f6feb;
      color: white;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 0.85em;
      margin-right: 8px;
    }}
    .content {{
      background: #0d1117;
      padding: 10px 0;
    }}
    footer {{
      margin-top: 40px;
      color: #484f58;
      font-size: 0.85em;
      text-align: center;
      border-top: 1px solid #21262d;
      padding-top: 20px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>AI Market Analysis</h1>
    <span class="badge">Claude Opus 4.6</span>
    <span class="badge">{market}</span>
    <span class="badge">{now}</span>
  </div>
  <div class="content">
    <p>{html_body}</p>
  </div>
  <footer>
    Generated by Claude Market Analyst — Sector Rotation &amp; Fund Flow Analyzer
  </footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [SAVED] {output_path}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Claude AI Market Analyst",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--market", default="ALL", choices=["JP", "US", "ALL"],
        help="Market to analyze (default: ALL)",
    )
    parser.add_argument(
        "--screener-dir",
        default=os.path.dirname(__file__),
        help="Path to screener/ directory with CSV outputs",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    parser.add_argument(
        "--export-html",
        action="store_true",
        help="Export analysis to HTML file",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Include backtest results in analysis",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick signal check only (uses Haiku, fast & cheap)",
    )

    args = parser.parse_args()

    # データ読み込み
    print(f"Loading market data from {args.screener_dir}...")
    market_data = load_market_data(args.screener_dir, args.market)

    # データ確認
    has_data = any(
        d["sector_metrics"] is not None or d["trade_candidates"] is not None
        for d in market_data.values()
    )
    if not has_data:
        print("[ERROR] 分析データが見つかりません。")
        print("  先に sector_flow_analyzer.py を実行してください:")
        print("  python3 screener/sector_flow_analyzer.py screener/snapshots/ --export-csv")
        sys.exit(1)

    # バックテスト結果
    backtest_data = None
    if args.backtest:
        mkt = "US" if args.market == "ALL" else args.market
        backtest_data = load_backtest_results(args.screener_dir, mkt)
        if backtest_data:
            print(f"  Backtest data loaded (trades={backtest_data['total_trades']})")

    # アナリスト初期化
    try:
        analyst = ClaudeMarketAnalyst(
            api_key=args.api_key,
            model=args.model,
            use_streaming=not args.no_stream,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # クイックモード
    if args.quick:
        summary_parts = []
        for mkt, d in market_data.items():
            if d["sector_metrics"] is not None:
                sm = d["sector_metrics"]
                summary_parts.append(f"=== {mkt} Sector Metrics ===\n{_df_to_compact_str(sm, 10)}")
        summary = "\n".join(summary_parts)
        print("\n[Quick Signal Check]")
        result = analyst.quick_signal_check(summary)
        print(result)
        return

    # フルアナリシス
    prompt = build_analysis_prompt(market_data, backtest_data)
    analysis = analyst.analyze(prompt)

    # HTMLエクスポート
    if args.export_html and analysis:
        html_path = os.path.join(
            args.screener_dir,
            f"ai_analysis_{args.market}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        )
        export_html_analysis(analysis, html_path, args.market)

    print("\n" + "=" * 60)
    print("分析完了。")
    if args.export_html:
        print(f"  HTMLレポート: screener/ai_analysis_{args.market}_*.html")


if __name__ == "__main__":
    main()

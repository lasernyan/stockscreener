#!/usr/bin/env python3
"""
COWORK Runner — Claude Code Agent による自律的な市場分析
=========================================================
claude_agent_sdk を使い、Claude Code が直接ファイルシステムを操作して
データ取得・セクター分析・バックテスト・レポート生成を自律的に実行する。

Anthropic API (claude_analyst.py) との違い:
  - claude_analyst.py  : Python → Anthropic API → トークン課金
  - cowork_runner.py   : Python → Claude Code CLI → Pro/Max サブスクリプション消費
                         Claude 自身が Bash/Read/Write ツールでファイルを直接操作

Requirements:
  pip install claude-agent-sdk
  Claude Code CLI がインストール・ログイン済みであること

Usage:
  python3 screener/cowork_runner.py
  python3 screener/cowork_runner.py --no-fetch      # データ取得スキップ
  python3 screener/cowork_runner.py --no-backtest   # バックテストスキップ
  python3 screener/cowork_runner.py --market JP     # 日本市場のみ
  python3 screener/cowork_runner.py --max-turns 30  # エージェントターン数上限
"""

import argparse
import os
import sys
from datetime import datetime

try:
    import anyio
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        CLIConnectionError,
        CLINotFoundError,
        ResultMessage,
        SystemMessage,
        query,
    )
except ImportError:
    print("[ERROR] claude-agent-sdk が必要です:")
    print("  pip install claude-agent-sdk")
    sys.exit(1)

# ============================================================
# エージェントプロンプト
# ============================================================

SYSTEM_PROMPT = """あなたは機関投資家レベルの市場アナリスト兼クオンツエンジニアです。
提供されたツールを使ってセクターローテーション分析を実行し、
実践的かつ具体的な投資レポートを日本語で生成してください。

分析の際は必ず以下を含めてください：
1. マクロ的なリスクオン/リスクオフ判断
2. 資金フローの方向と強度
3. セクターローテーションのフェーズ
4. 暗号資産・コモディティ・FXとの相互関係（データがあれば）
5. 具体的なアクションプラン（エントリー条件・リスク管理込み）
"""


def build_analysis_prompt(
    project_dir: str,
    market: str,
    run_fetch: bool,
    run_backtest: bool,
) -> str:
    """Claude Code エージェントへの指示プロンプトを構築"""

    screener_dir = os.path.join(project_dir, "screener")
    snapshots_dir = os.path.join(screener_dir, "snapshots")
    now_str = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = os.path.join(screener_dir, f"ai_analysis_cowork_{market}_{now_str}.html")

    market_opts = ""
    if market != "ALL":
        market_opts = f"--market {market}"

    steps = []

    # Step 0: データ取得
    if run_fetch:
        steps.append(f"""
## Step 0: マルチアセットデータ取得
以下のコマンドを実行してください:
```bash
cd {project_dir}
python3 screener/data_fetcher.py --market ALL --output-dir screener/snapshots/
```
エラーが出ても続行してください（一部のティッカーは取得できない場合があります）。
""")

    # Step 1: セクターフロー分析
    steps.append(f"""
## Step 1: セクターフロー分析
以下のコマンドを実行してください:
```bash
cd {project_dir}
python3 screener/sector_flow_analyzer.py screener/snapshots/ --top 5 --export-csv --html
```
""")

    # Step 2: バックテスト
    if run_backtest:
        bt_market = "US" if market == "ALL" else market
        steps.append(f"""
## Step 2: バックテスト実行
以下のコマンドを実行してください:
```bash
cd {project_dir}
python3 screener/backtester.py --snapshot-dir screener/snapshots/ --market {bt_market} --top-long 3 --export-html --export-csv
```
""")

    # Step 3: 生成されたCSVを読み込んで分析
    steps.append(f"""
## Step 3: 生成データの読み込み
以下のファイルを読み込んで内容を把握してください:
- screener/sector_metrics_us.csv  (存在すれば)
- screener/sector_metrics_jp.csv  (存在すれば)
- screener/trade_candidates_us.csv (存在すれば)
- screener/trade_candidates_jp.csv (存在すれば)
- screener/backtest_trades_*.csv   (存在すれば)

Glob ツールで `screener/sector_metrics*.csv` と `screener/trade_candidates*.csv` を検索してから Read してください。
""")

    # Step 4: レポート生成
    steps.append(f"""
## Step 4: 分析レポートの生成
上記データを基に以下の構成で詳細な日本語分析レポートを作成し、
`{report_path}` に書き込んでください。

### レポート構成 (HTMLファイルとして保存):
1. **市場全体サマリー** — リスクオン/オフ状態、主要テーマ (3-5文)
2. **セクター分析** — SCSスコアの意味、注目すべき変化、ローテーションのフェーズ
3. **資金フロー分析** — どこに資金が流入/流出しているか
4. **ロング戦略 TOP3** — 銘柄名・エントリー条件・目標・損切りを明記
5. **ショート戦略 TOP3** — 同様に条件付き
6. **バックテスト評価** — シグナルの統計的有意性、改善提案
7. **リスクシナリオ** — 現在の分析が崩れる条件 (2-3個)
8. **今週のアクションプラン** — 具体的かつ実行可能なもの

### HTML形式の要件:
- ダークテーマ (背景: #0d1117, テキスト: #c9d1d9)
- セクションごとに見やすく構造化
- 重要な数値はハイライト表示
- 生成日時を明記

ファイルパス: {report_path}
""")

    prompt = f"""# 市場分析タスク
日時: {datetime.now().strftime('%Y-%m-%d %H:%M JST')}
対象市場: {market}
プロジェクトディレクトリ: {project_dir}

以下のステップを順番に実行し、最後に詳細な市場分析レポートを生成してください。
各ステップでエラーが出た場合は記録して次のステップに進んでください。

{''.join(steps)}

## 最終確認
すべてのステップが完了したら:
1. 生成したHTMLファイルのパスを報告してください
2. 分析の主要ポイントを箇条書きで3点まとめてください
3. 次回分析への改善提案があれば1点述べてください
"""
    return prompt


# ============================================================
# エージェント実行
# ============================================================

async def run_cowork_agent(
    project_dir: str,
    market: str,
    run_fetch: bool,
    run_backtest: bool,
    max_turns: int,
    verbose: bool,
) -> str:
    """Claude Code エージェントを起動して分析を実行"""

    prompt = build_analysis_prompt(project_dir, market, run_fetch, run_backtest)

    if verbose:
        print(f"\n[COWORK] Claude Code エージェント起動...")
        print(f"  project_dir : {project_dir}")
        print(f"  market      : {market}")
        print(f"  fetch       : {run_fetch}")
        print(f"  backtest    : {run_backtest}")
        print(f"  max_turns   : {max_turns}")
        print("-" * 60)

    result_text = ""
    session_id = None

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=project_dir,
                allowed_tools=["Bash", "Read", "Write", "Glob", "Grep"],
                permission_mode="acceptEdits",   # ファイル書き込みを自動承認
                max_turns=max_turns,
                system_prompt=SYSTEM_PROMPT,
            ),
        ):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
                if verbose:
                    print(f"  session_id  : {session_id}")

            elif isinstance(message, ResultMessage):
                result_text = message.result
                if verbose:
                    print("\n" + "=" * 60)
                    print("[COWORK] 分析完了")
                    print("=" * 60)
                    print(result_text)

    except CLINotFoundError:
        print("\n[ERROR] Claude Code CLI が見つかりません。")
        print("  インストール: pip install claude-agent-sdk")
        print("  または: https://claude.ai/download からインストール")
        raise
    except CLIConnectionError as e:
        print(f"\n[ERROR] Claude Code CLI 接続エラー: {e}")
        raise

    return result_text


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="COWORK Runner — Claude Code Agent による自律的な市場分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python3 screener/cowork_runner.py
  python3 screener/cowork_runner.py --market JP --no-fetch
  python3 screener/cowork_runner.py --max-turns 40
        """,
    )
    parser.add_argument(
        "--market", default="ALL", choices=["JP", "US", "ALL"],
        help="分析対象市場 (default: ALL)",
    )
    parser.add_argument(
        "--project-dir",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="プロジェクトルートディレクトリ",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="データ取得をスキップ（既存データで分析）",
    )
    parser.add_argument(
        "--no-backtest", action="store_true",
        help="バックテストをスキップ",
    )
    parser.add_argument(
        "--max-turns", type=int, default=35,
        help="エージェントの最大ターン数 (default: 35)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="詳細ログを抑制",
    )

    args = parser.parse_args()

    print(f"COWORK Runner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Claude Code Agent SDK による自律分析")
    print(f"  ※ Claude Code Pro/Max サブスクリプションを使用します")
    print(f"  ※ Anthropic API トークンは消費しません")

    try:
        anyio.run(
            run_cowork_agent,
            args.project_dir,
            args.market,
            not args.no_fetch,
            not args.no_backtest,
            args.max_turns,
            not args.quiet,
        )
    except (CLINotFoundError, CLIConnectionError):
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] 中断しました。")
        sys.exit(0)


if __name__ == "__main__":
    main()

# Sector Rotation & Fund Flow Analyzer v2

トップダウン・アプローチによるセクター強弱分析ツール。

```
市場全体 → セクター強弱(SCS) → 資金フロー → ローテーション検出 → 個別銘柄シグナル
```

## 5つの分析レイヤー

| レイヤー | 概要 |
|---|---|
| **Sector Composite Score (SCS)** | 5指標を加重合成した0-100のセクター強弱スコア |
| **Estimated Fund Flow (EFF)** | `Σ(Volume × Change% / 100)` で実質資金流入/流出を推定 |
| **Market Breadth** | SMA200上の銘柄比率でセクター健康状態を評価 |
| **Rotation Detection** | スナップショット比較によるローテーション検出 |
| **Trade Signal Generation** | 強セクター内のLONG / 弱セクター内のSHORT候補を抽出 |

### SCS 重み付け

| 要素 | Weight | 内容 |
|---|---|---|
| Momentum | 25% | セクター内銘柄の平均変動率 |
| Trend | 20% | RSI + MACD乖離の合成 |
| Breadth | 25% | SMA200上の銘柄比率（最重要） |
| Fund Flow | 20% | 出来高加重の資金流入/流出 |
| ADX | 10% | トレンド明確度 |

## データソース

| ソース | 用途 | API Key |
|---|---|---|
| **yfinance** | 株価・テクニカル指標 | 不要 |
| **EDINET API** | 公開買付・大量保有報告・自社株買い | `EDINET_API_KEY` |
| **e-STAT API** | 景気動向・生産指数などマクロ指標 | `ESTAT_APP_ID` |

## セットアップ

```bash
cd stockanalyzeV2
pip install -r requirements.txt
```

### API Keys（任意）

```bash
export EDINET_API_KEY="your_key_here"   # https://disclosure.edinet-fsa.go.jp
export ESTAT_APP_ID="your_app_id"       # https://www.e-stat.go.jp/api
```

## 使い方

### CLI

```bash
# 日本市場の分析（デフォルト）
python -m stockanalyzeV2 --market JP --top 5 --export-csv

# 米国市場の分析
python -m stockanalyzeV2 --market US --top 10

# EDINET・e-STAT を含めた完全分析
python -m stockanalyzeV2 --market JP \
  --edinet-key $EDINET_API_KEY \
  --estat-id   $ESTAT_APP_ID  \
  --export-csv --top 10

# ヘルプ
python -m stockanalyzeV2 --help
```

### Python API

```python
from stockanalyzeV2.analyzer import SectorAnalyzer

# 基本分析
analyzer = SectorAnalyzer(market="JP", top_n=5)
result   = analyzer.run(export_csv=True)

# ローテーション検出付き（前回スナップショットと比較）
prev_result = SectorAnalyzer(market="JP").run(print_report=False)
analyzer    = SectorAnalyzer(market="JP", previous_metrics=prev_result.sector_metrics)
result      = analyzer.run()

# 結果アクセス
for sm in result.top_sectors(3):
    print(f"{sm.sector}: SCS={sm.scs:.1f}  Breadth={sm.pct_above_sma200:.0%}")

for c in result.long_candidates[:5]:
    print(f"LONG  {c.ticker}  score={c.total_score:.0f}  rsi={c.rsi:.1f}")

for c in result.short_candidates[:5]:
    print(f"SHORT {c.ticker}  score={c.total_score:.0f}  rsi={c.rsi:.1f}")
```

## ディレクトリ構成

```
stockanalyzeV2/
├── __init__.py
├── __main__.py          # CLI エントリポイント
├── analyzer.py          # メインオーケストレーター
├── config.py            # セクター定義・閾値設定
├── requirements.txt
├── data_fetchers/
│   ├── yfinance_fetcher.py   # 株価・テクニカル指標取得
│   ├── edinet_fetcher.py     # EDINET API
│   └── estat_fetcher.py      # e-STAT API
├── analysis/
│   ├── scs_calculator.py     # SCS 計算
│   ├── rotation_detector.py  # ローテーション検出
│   ├── signal_generator.py   # トレードシグナル生成
│   └── reporter.py           # コンソール・CSV レポート
└── models/
    └── sector_models.py      # データクラス定義
```

## 出力ファイル（--export-csv 時）

| ファイル | 内容 |
|---|---|
| `sector_metrics_{MARKET}_{timestamp}.csv` | セクター別SCS・指標 |
| `trade_candidates_{MARKET}_{timestamp}.csv` | LONG/SHORTシグナル一覧 |

## LONG / SHORT シグナル条件

### LONG（強セクター内、SCS ≥ 60）
- Close > SMA50（中期上昇トレンド）
- RSI 40–70（過熱でないモメンタム）
- Relative Volume > 1.0（出来高増加）
- MACD > Signal（上昇モメンタム）
- ADX > 20（トレンド明確）
- ローテーション流入中 → ボーナス加点 (+5)

### SHORT（弱セクター内、SCS ≤ 40）
- Close < SMA50（中期下降トレンド）
- RSI 30–60（まだ売り余地）
- MACD < Signal（下降モメンタム）
- ADX > 20（トレンド明確）
- ローテーション流出中 → ボーナス加点 (+5)

## Market Breadth 判定

| SMA200上銘柄比率 | 判定 |
|---|---|
| > 80% | Very Strong（上昇トレンドが広範囲） |
| 60–80% | Strong |
| 40–60% | Neutral |
| 20–40% | Weak |
| < 20% | Very Weak（下降トレンドが支配的） |

## カスタマイズ

`config.py` でセクター銘柄リスト・SCS重み・閾値をすべて変更できます。

```python
# SCSWeights のカスタマイズ例
from stockanalyzeV2.config import SCSWeights
from stockanalyzeV2.analyzer import SectorAnalyzer

weights = SCSWeights(momentum=0.30, breadth=0.35, trend=0.15, fund_flow=0.15, adx=0.05)
analyzer = SectorAnalyzer(market="JP", weights=weights)
```

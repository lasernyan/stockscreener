# Sector Rotation & Fund Flow Strategy

TradingView スクリーナーの CSV エクスポートを使い、**セクター間の資金ローテーション** を検出して売買候補を自動抽出する戦略ツール。

## コンセプト

個別銘柄ではなく **セクター全体の健康状態** を先に評価し、強いセクターからロング候補、弱いセクターからショート候補を絞り込む「トップダウン・アプローチ」。

```
市場全体 → セクター強弱 → 資金フロー方向 → ローテーション検出 → 個別銘柄選定
```

## 5つの分析レイヤー

### 1. Sector Composite Score (SCS)
複数指標を加重合成した0-100のセクター強弱スコア。

| 要素 | Weight | 内容 |
|------|--------|------|
| Momentum | 25% | セクター内銘柄の平均変動率 |
| Trend | 20% | RSI + MACD乖離の合成 |
| Breadth | 25% | SMA200上の銘柄比率 (最重要) |
| Fund Flow | 20% | 出来高加重の資金流入/流出 |
| ADX | 10% | トレンド明確度 |

### 2. Estimated Fund Flow (EFF)
```
Fund Flow = Σ (Volume × Change% / 100)
```
出来高を伴う価格変動だけが「本物の資金移動」であるという仮定に基づく。

### 3. Market Breadth
セクター内で各移動平均を上回る銘柄の比率:
- **SMA200上 > 80%** → Very Strong (上昇トレンドが広範囲)
- **SMA200上 40-60%** → Neutral (方向性不明確)
- **SMA200上 < 20%** → Very Weak (下降トレンドが支配的)

**ブレッスダイバージェンス**: セクター指数が上昇しているのにBreadthが低下している場合、上昇の持続性に疑問。

### 4. Rotation Detection
2つ以上のスナップショットを比較し、SCSの変化量が閾値を超えたセクターを検出:
- **Rising Sector**: SCS が +3以上上昇
- **Falling Sector**: SCS が -3以上下降
- **Rotation Pair**: Falling → Rising のペア (資金移動の証拠)

### 5. Trade Signal Generation

**LONG 条件** (強セクター内):
- Close > SMA50 (中期上昇トレンド)
- RSI 40-70 (過熱でないモメンタム)
- Relative Volume > 1.0 (出来高増加)
- MACD > Signal (上昇モメンタム)
- ADX > 20 (トレンド明確)
- ローテーション流入中 → ボーナス加点

**SHORT 条件** (弱セクター内):
- Close < SMA50 (中期下降トレンド)
- RSI 30-60 (まだ売り余地)
- MACD < Signal (下降モメンタム)
- ローテーション流出中 → ボーナス加点

## 使い方

### Step 1: TradingView からCSVエクスポート

1. [TradingView Screener](https://www.tradingview.com/screener/) を開く
2. フィルタ設定: US Market / Market Cap > $10B
3. 表示カラム:
   - Ticker, Name, Sector, Industry
   - Close, Change%, Volume, Relative Volume, Market Cap
   - SMA20, SMA50, SMA200
   - RSI, MACD, MACD Signal, ADX, ATR
   - P/E, EPS (任意)
4. 「Export」で CSV ダウンロード

### Step 2: ファイル配置
```bash
screener/snapshots/screener_2025-01-06.csv
screener/snapshots/screener_2025-01-07.csv
# ファイル名に日付 (YYYY-MM-DD) を含める
```

### Step 3: 分析実行
```bash
# 基本実行
python screener/sector_flow_analyzer.py screener/snapshots/

# チャート出力先を指定
python screener/sector_flow_analyzer.py screener/snapshots/ --out my_report.png

# CSV も出力
python screener/sector_flow_analyzer.py screener/snapshots/ --export-csv

# ショートスクリプト
chmod +x screener/run_analysis.sh
./screener/run_analysis.sh --export-csv
```

## 出力

### ターミナルレポート
- セクターランキング (SCS, Change%, RSI, ADX, Breadth, FundFlow)
- 資金フローサマリー
- マーケットブレッス (MA上の銘柄比率 + 健全性判定)
- ローテーションシグナル (上昇/下降セクター、資金移動ペア)
- 売買候補リスト (理由付き)

### ダッシュボードチャート (`sector_dashboard.png`)
6パネル構成:
1. Sector Composite Score (横棒グラフ)
2. 推定資金フロー (横棒グラフ)
3. Market Breadth (MA別のグループ棒グラフ)
4. RSI vs ADX バブルチャート (色=SCS, サイズ=時価総額)
5. SCS 時系列推移 (複数スナップショット時)
6. 売買候補テーブル

### CSV出力
- `sector_metrics.csv` — セクター別の全指標
- `trade_candidates.csv` — 売買候補と理由

## 日次ワークフロー例

```
朝: TradingView で CSV エクスポート → snapshots/ に保存
     ↓
     ./screener/run_analysis.sh --export-csv
     ↓
確認: セクターランキングの変化を前日と比較
     ↓
判断: ローテーションシグナルに基づき売買候補を評価
     ↓
実行: 個別銘柄のチャートを TradingView で最終確認 → エントリー
```

## Claude Code との連携アイデア

スナップショットを蓄積すれば、Claude Code に以下の指示を出して分析を深化できる:

- 「直近5日分の CSVからローテーション傾向を要約して」
- 「Technology セクターの breadth が低下しているのに SCS が高いのはなぜ？」
- 「エネルギーセクターの Fund Flow が急減した理由を trade_candidates.csv から考察して」
- 「sector_metrics.csv から週次のヒートマップを作って」

## 依存関係

```
pip install pandas numpy matplotlib
```

## ファイル構成

```
screener/
├── STRATEGY.md                  # この文書
├── sector_flow_analyzer.py      # コアエンジン
├── run_analysis.sh              # ランナースクリプト
├── snapshots/                   # TradingView CSV を置く場所
│   ├── example_2025-01-06.csv   # サンプルデータ
│   └── example_2025-01-07.csv   # サンプルデータ
├── sector_dashboard.png         # 生成されるチャート
├── sector_metrics.csv           # 生成されるセクター指標
└── trade_candidates.csv         # 生成される売買候補
```

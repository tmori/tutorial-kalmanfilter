# カルマンフィルタ教育ツール

## 概要
このプログラムは、ノイズのある時系列信号に対して**カルマンフィルタ**を適用し、ノイズを低減する方法をデモンストレーションします。合成されたノイズ付きデータを生成し、シンプルなカルマンフィルタを適用することで、フィルタがどのようにノイズを減らし、基になる信号を再構築するかを観察できます。このプログラムは、カルマンフィルタの理論的基盤と実用的な経験を提供することを目的としています。

---

## カルマンフィルタとは？
**カルマンフィルタ**は、動的システムの状態をノイズの多い観測データから推定するためのアルゴリズムです。制御システム、ロボティクス、経済学、ナビゲーションなどの分野で広く使用されています。

カルマンフィルタは主に以下の2つのステップで動作します：
1. **予測（Prediction）:** システムモデルに基づいて現在の状態を予測します。
2. **更新（Update）:** 新しい観測データを使用して推定値を調整し、モデル予測と観測データのバランスを取ります。

主な利点：
- ガウスノイズを伴う線形システムに対して最適。
- 計算負荷が低い。

---

## アルゴリズムの理論的背景
カルマンフィルタは、システムの状態 \( x \) と、それに関連する不確実性（共分散行列 \( P \) で表される）を推定します。この値は以下のステップで繰り返し更新されます：

1. **予測ステップ:**
   - 現在の推定値に基づいて次の状態を予測：
     \[
     \hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B u_k
     \]
     - \( A \): 状態遷移行列
     - \( B \): 制御入力行列
     - \( u_k \): 制御ベクトル
   - 誤差共分散を予測：
     \[
     P_{k|k-1} = A P_{k-1|k-1} A^T + Q
     \]
     - \( Q \): プロセスノイズ共分散

2. **更新ステップ:**
   - カルマンゲインを計算：
     \[
     K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
     \]
     - \( H \): 観測行列
     - \( R \): 観測ノイズ共分散
   - 推定値を更新：
     \[
     \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1})
     \]
     - \( z_k \): 観測ベクトル
   - 誤差共分散を更新：
     \[
     P_{k|k} = (I - K_k H) P_{k|k-1}
     \]

このプログラムでは、1次元の状態推定に焦点を当てたシンプルなスカラー版カルマンフィルタを実装しています。

---

## 教育的な目標
このプログラムは以下を目指します：
1. **カルマンフィルタの入門:** 信号のノイズ除去という文脈で、予測と更新の基本概念を体験します。
2. **実用的なケースを示す:** ノイズのあるデータにカルマンフィルタを適用して基になる信号を復元する方法を示します。
3. **結果の視覚化:** 生のノイズデータ、真の信号、フィルタ後の信号を明確に比較するグラフを提供します。
4. **簡潔な実装:** 初学者に適した最小限で実用的なカルマンフィルタアルゴリズムを提示します。

---

## プログラムの動作
1. **ノイズ付きデータの生成:**
   ユーザー定義の関数（例：正弦波）に基づいて合成時系列データを生成します。ガウスノイズを追加してノイズ観測をシミュレーションします。

2. **カルマンフィルタの適用:**
   ノイズ付きデータにカルマンフィルタを適用します。以下のパラメータを設定可能：
   - **プロセス分散:** システムモデルへの信頼度を決定。
   - **観測分散:** 観測値への信頼度を決定。

3. **結果の出力:**
   - 真の信号、ノイズ付き観測、フィルタ後の信号を含むCSVファイル（`filtered_data.csv`）を生成します。
   - 信号を比較するプロットを表示し、フィルタの有効性を示します。

---

## 使用方法
1. 必要なライブラリをインストール：
   ```bash
   pip install numpy pandas matplotlib
   ```
2. プログラムを実行：
   ```bash
   python kalman_filter_tool.py
   ```
3. `main()`関数内のパラメータを調整して、異なるノイズレベルやフィルタ設定を試してください。

---

## 教育的な学び
- **ノイズ低減:** カルマンフィルタがノイズの多い信号をどのように平滑化するかを観察。
- **パラメータ調整:** プロセス分散と観測分散の影響を理解。
- **実世界での応用:** ナビゲーションシステム、金融、ロボティクスでのカルマンフィルタの使用例を学ぶ。

---

## 制限と次のステップ
- **簡易実装:** このプログラムはスカラーカルマンフィルタを実装しており、1次元データに最適です。多次元システムへの拡張には行列操作が必要です。
- **動的モデル:** 時変システムの状態遷移モデルを導入。
- **拡張カルマンフィルタ (EKF):** この実装を拡張して非線形システムを探索。

---

## 参考文献
1. [Kalman Filter Tutorial - Welch & Bishop](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
2. [Wikipedia: Kalman Filter](https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AB%E3%83%9E%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF)
3. Simon Haykin, "Kalman Filtering and Neural Networks"


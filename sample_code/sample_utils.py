import os
import sys
import pandas as pd
import numpy as np
import time

# プロジェクトルートをパスに追加してモジュールをインポートできるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.comparison.cosine_similarity import cosine_similarity, flatten_self_sim_df
from utils.comparison.entropy import get_entropy, get_entropy_all
from utils.comparison.js_divergence import get_js_divergence_at_once
from utils.threading_process import run_in_executor


def sample_threading():
    print("\n--- Threading Sample ---")

    def heavy_task(name, duration):
        print(f"Task {name} started")
        time.sleep(duration)
        print(f"Task {name} finished")
        return f"Result {name}"

    try:
        # run_in_executor を使用して並行実行
        future1 = run_in_executor(heavy_task, "A", 2)
        future2 = run_in_executor(heavy_task, "B", 1)

        print("Waiting for tasks to complete...")
        res1 = future1.result()
        res2 = future2.result()
        print(f"Got results: {res1}, {res2}")
    except Exception as e:
        print(f"Error in threading sample: {e}")


def sample_comparison():
    print("\n--- Comparison Utils Sample ---")

    # データの準備
    matrix_a = pd.DataFrame(
        np.random.rand(5, 3), index=[f"item_a_{i}" for i in range(5)]
    )
    matrix_b = pd.DataFrame(
        np.random.rand(4, 3), index=[f"item_b_{i}" for i in range(4)]
    )

    print("Matrix A shape:", matrix_a.shape)
    print("Matrix B shape:", matrix_b.shape)

    # 1. コサイン類似度
    print("\n[Cosine Similarity]")
    cos_sim = cosine_similarity(matrix_a, matrix_b)
    print(cos_sim)

    # コサイン類似度の自己類似度とフラット化
    print("\n[Cosine Similarity (Self & Flatten)]")
    self_sim = cosine_similarity(matrix_a, matrix_a)
    flattened = flatten_self_sim_df(self_sim)
    print("Flattened self similarity (upper triangle):", flattened)

    # 2. エントロピー
    print("\n[Entropy]")
    # 単一リストのエントロピー
    counts = [10, 20, 30, 40]
    entropy = get_entropy(counts)
    print(f"Entropy of {counts}: {entropy}")

    # 行列の各行のエントロピー
    print("\n[Entropy (All Rows)]")
    matrix_counts = np.array(
        [[10, 10, 10], [100, 0, 0], [5, 5, 5]]  # 偏りがある -> エントロピー低い
    )
    entropies = get_entropy_all(matrix_counts)
    print(f"Entropies of rows:\n{entropies}")

    # 3. Jensen-Shannon Divergence
    print("\n[JS Divergence]")
    # 確率分布として正規化されたデータを使用
    prob_a = matrix_a.div(matrix_a.sum(axis=1), axis=0)  # 行の合計を1にする
    prob_b = matrix_b.div(matrix_b.sum(axis=1), axis=0)

    js_div = get_js_divergence_at_once(prob_a, prob_b)
    print(js_div)


def main():
    sample_threading()
    sample_comparison()


if __name__ == "__main__":
    main()

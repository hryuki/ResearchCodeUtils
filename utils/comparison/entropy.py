import numpy as np
import pandas as pd

def get_entropy(num_list):
    """
    リストからエントロピーを計算（ベクトル計算版）。
    Args:
        num_list (list or np.ndarray): 数値のリストまたはNumPy配列。各要素は非負整数である必要があります。
    Returns:
        float: エントロピーの値。
    """
    # NumPy配列に変換
    if not isinstance(num_list, np.ndarray):
        counts = np.array(num_list)
    
    # 各要素の出現確率を一度に計算
    probabilities = counts / counts.sum()
    
    # 確率が0の要素は計算から除外（log2(0)はエラーになるため）
    probabilities = probabilities[probabilities > 0]
    
    # ベクトル計算でエントロピーを一括計算
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def get_entropy_all(matrix):
    """
    各行のエントロピーをまとめて計算する関数（完全ベクトル計算版）。
    
    Args:
        matrix (np.ndarray or pd.DataFrame): 各行が度数分布または確率分布を表す2次元配列。
        
    Returns:
        np.ndarray: 各行のエントロピーを格納した1次元配列。
    """
    # NumPy配列に変換
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    elif isinstance(matrix, list):
        # リストの長さが異なる場合、NumPy配列に変換できないため、値に影響が出ないように値を追加する
        max_length = max(len(row) for row in matrix)
        matrix = [row + [0] * (max_length - len(row)) for row in matrix]
        matrix = np.array(matrix)
    
    # 1. 各行の合計を計算 (形状を(n, 1)にしてブロードキャスト可能にする)
    row_sums = matrix.sum(axis=1, keepdims=True)
    
    # 2. 0による除算を回避
    #    行の合計が0の場合、確率はすべて0となり、エントロピーも0になる
    #    ここでは計算上、分母が0にならないように1に置き換える
    safe_row_sums = np.where(row_sums > 0, row_sums, 1)
    
    # 3. 各行の確率分布を一度に計算
    probabilities = matrix / safe_row_sums
    
    # 4. log(0)を回避しつつ、p * log2(p) を一度に計算
    #    確率が0の項はエントロピーへの寄与が0なので、計算結果も0にする
    term = np.zeros_like(probabilities, dtype=float)
    non_zero_mask = probabilities > 0
    term[non_zero_mask] = probabilities[non_zero_mask] * np.log2(probabilities[non_zero_mask])

    # 5. 各行の合計を計算してエントロピーを求める
    entropies = -np.sum(term, axis=1)
    
    return entropies
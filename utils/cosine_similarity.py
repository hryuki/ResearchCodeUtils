import pandas as pd
import numpy as np

def cosine_similarity(matrix_a, matrix_b) -> pd.DataFrame:
    """
    コサイン類似度を計算する関数
    Args:
        matrix_a (pd.DataFrame or np.ndarray): 入力行列A
        matrix_b (pd.DataFrame or np.ndarray): 入力行列B
    Returns:
        pd.DataFrame: コサイン類似度を含むデータフレーム
    """
    if not isinstance(matrix_a, pd.DataFrame):
        matrix_a = pd.DataFrame(matrix_a)
    if not isinstance(matrix_b, pd.DataFrame):
        matrix_b = pd.DataFrame(matrix_b)
    
    # 行列ベクトルの片方を転置して積を求める
    df_dot = matrix_a.dot(matrix_b.T)
    # 行列ノルムを求める
    matrix_a_norm = pd.DataFrame(np.linalg.norm(matrix_a.values, axis=1), index = matrix_a.index)
    matrix_b_norm = pd.DataFrame(np.linalg.norm(matrix_b.values, axis=1), index = matrix_b.index)
    # 行列ノルムの片方を転置して積を求める
    df_norm = matrix_a_norm.dot(matrix_b_norm.T)
    # コサイン類似度を算出
    df_cos = df_dot/df_norm
    # NaNを0に置き換える
    df_cos = df_cos.fillna(0)
    return df_cos

def flatten_self_sim_df(df_cos: pd.DataFrame) -> np.ndarray:
    """
    cosine_similarity(df, df)のように自己類似度を計算した結果のデータフレームをフラットな形式に変換する関数
    これは、上三角行列の値を1次元の配列に変換するために使用されます。
    Args:
        df_cos (pd.DataFrame): コサイン類似度のデータフレーム
    Returns:
        np.ndarray: フラットな形式のnp.ndarray
    """
    return df_cos.values[np.triu_indices_from(df_cos.values, k=1)]


if __name__ == "__main__":
    # Example usage
    matrix_a = pd.DataFrame(np.random.rand(5, 3), index=[f'item_{i}' for i in range(5)])
    matrix_b = pd.DataFrame(np.random.rand(4, 3), index=[f'item_{i}' for i in range(4)])
    result = cosine_similarity(matrix_a, matrix_b)
    print(result)
    
    # Example usage 2
    matrix_a = np.random.rand(5, 3)
    matrix_b = np.random.rand(4, 3)
    result = cosine_similarity(matrix_a, matrix_b)
    print(result)
    
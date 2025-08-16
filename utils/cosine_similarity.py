import pandas as pd

def cos_sim(matrix_a, matrix_b) -> pd.DataFrame:
    """
    コサイン類似度を計算する関数
    Args:
        matrix_a (pd.DataFrame or np.ndarray): 入力行列A
        matrix_b (pd.DataFrame or np.ndarray): 入力行列B
    Returns:
        pd.DataFrame: コサイン類似度を含むデータフレーム
    """
    
    import pandas as pd
    import numpy as np
    
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

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    # Example usage
    matrix_a = pd.DataFrame(np.random.rand(5, 3), index=[f'item_{i}' for i in range(5)])
    matrix_b = pd.DataFrame(np.random.rand(4, 3), index=[f'item_{i}' for i in range(4)])
    result = cos_sim(matrix_a, matrix_b)
    print(result)
    
    # Example usage 2
    matrix_a = np.random.rand(5, 3)
    matrix_b = np.random.rand(4, 3)
    result = cos_sim(matrix_a, matrix_b)
    print(result)
    
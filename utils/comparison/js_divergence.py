import numpy as np
import pandas as pd


def get_KL_divergence(p, q):
    return sum([p[i] * np.log(p[i] / q[i]) for i in range(len(p)) if p[i] != 0])


def get_JS_divergence(p, q):
    m = (p + q) / 2
    return 0.5 * get_KL_divergence(p, m) + 0.5 * get_KL_divergence(q, m)


def get_js_divergence_at_once(matrix_a, matrix_b):
    """
    2つのデータフレームに含まれる全確率分布ペアのJSダイバージェンスを計算する関数。

    Args:
        matrix_a (pd.DataFrame or np.ndarray): 確率分布を各行に格納したデータフレームまたはNumpy配列。
        matrix_b (pd.DataFrame or np.ndarray): 確率分布を各行に格納したデータフレームまたはNumpy配列。

    Returns:
        pd.DataFrame: matrix_aの各行とmatrix_bの各行の間のJSダイバージェンスを格納したデータフレーム。
    """
    # 入力がNumpy配列の場合、データフレームに変換
    if not isinstance(matrix_a, pd.DataFrame):
        matrix_a = pd.DataFrame(matrix_a)
    if not isinstance(matrix_b, pd.DataFrame):
        matrix_b = pd.DataFrame(matrix_b)

    # DataFrameをNumpy配列に変換
    p = matrix_a.values
    q = matrix_b.values

    # ブロードキャストのために次元を追加
    # p: (n, d) -> (n, 1, d)
    # q: (m, d) -> (1, m, d)
    p_exp = p[:, np.newaxis, :]
    q_exp = q[np.newaxis, :, :]

    # 中間分布 m を計算
    # m の shape は (n, m, d) となる
    m = (p_exp + q_exp) / 2

    # KLダイバージェンス D_KL(P||M) と D_KL(Q||M) を計算
    # np.log(0) による警告を抑制し、計算の安定性を確保
    with np.errstate(divide='ignore', invalid='ignore'):
        # P * log(P/M)
        kl_p_m_terms = p_exp * np.log(p_exp / m)
        # Q * log(Q/M)
        kl_q_m_terms = q_exp * np.log(q_exp / m)

    # PやQが0の要素は nan になるため、0に置換
    kl_p_m_terms[np.isnan(kl_p_m_terms)] = 0
    kl_q_m_terms[np.isnan(kl_q_m_terms)] = 0

    # 最後の次元(分布の各要素)で和を取り、KLダイバージェンスを算出
    kl_p_m = np.sum(kl_p_m_terms, axis=2)
    kl_q_m = np.sum(kl_q_m_terms, axis=2)

    # JSダイバージェンスを計算
    js_divergence = 0.5 * kl_p_m + 0.5 * kl_q_m

    # 結果をデータフレームに格納して返す
    return pd.DataFrame(js_divergence, index=matrix_a.index, columns=matrix_b.index)
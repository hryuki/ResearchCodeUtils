import os
import sys

# プロジェクトルートをパスに追加してモジュールをインポートできるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from openai_api.embedding import Embedding


def main():
    # APIキーの設定
    # 環境変数から取得するか、直接指定してください
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        # api_key = "your_api_key_here" # 直接指定する場合
        return

    # Embeddingクラスのインスタンス化
    embedder = Embedding(api_key=api_key)

    # テキストデータの準備
    texts = [
        "This is a sample sentence.",
        "Another example text for embedding.",
        "OpenAI API is powerful.",
    ]

    print(f"Input texts: {texts}")

    try:
        # 埋め込みベクトルの取得
        # model, retry_num, sleep_time は省略可能（デフォルト値が使用される）
        embeddings = embedder.embed(texts=texts, model="text-embedding-3-small")

        print(f"Number of embeddings returned: {len(embeddings)}")
        print(f"Dimension of the first embedding: {len(embeddings[0])}")

        # 次元削減の例 (デフォルトは256次元)
        reduced_embeddings = embedder.dimension_reduction(embeddings, dimension=128)
        print(f"Dimension after reduction: {len(reduced_embeddings[0])}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

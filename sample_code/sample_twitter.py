import os
import sys

# プロジェクトルートをパスに追加してモジュールをインポートできるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from twitter_api.twitter_api import TwitterAPI


def main():
    # Bearer Tokenの設定
    # 環境変数から取得するか、直接指定してください
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

    # ユーザー名は任意ですが、一部のAPI呼び出しで必要になる場合があります
    username = "example_user"

    if not bearer_token:
        print("Error: TWITTER_BEARER_TOKEN environment variable is not set.")
        # bearer_token = "your_bearer_token_here" # 直接指定する場合
        return

    # TwitterAPIクラスのインスタンス化
    twitter = TwitterAPI(bearer_token=bearer_token, username=username)

    # 1. クエリからツイートを検索する例
    query = "OpenAI -is:retweet"  # "OpenAI" を含む、リツイートではないツイートを検索
    print(f"Searching tweets with query: {query}")
    try:
        tweets = twitter.get_tweets_from_query(query=query, max_results=10)
        print(f"Found tweets: {tweets}")
    except Exception as e:
        print(f"Error searching tweets: {e}")

    # 2. ユーザーIDからツイートを取得する例 (要: ユーザーID)
    # user_id = "1234567890"
    # try:
    #     user_tweets = twitter.get_tweets_from_user_id(user_id=user_id, max_results=5)
    #     print(f"User tweets: {user_tweets}")
    # except Exception as e:
    #     print(f"Error getting user tweets: {e}")

    # 3. リツイートユーザーを取得する例 (要: ツイートID)
    # tweet_id = "1234567890123456789"
    # try:
    #     retweeters = twitter.get_retweet_user_from_tweet_id(tweet_id=tweet_id)
    #     print(f"Retweeters: {retweeters}")
    # except Exception as e:
    #     print(f"Error getting retweeters: {e}")


if __name__ == "__main__":
    main()

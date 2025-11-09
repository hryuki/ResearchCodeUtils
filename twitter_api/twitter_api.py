from typing import Literal
import requests
import datetime

class TwitterAPI:
    def __init__(self, bearer_token, username):
        self.bearer_token = bearer_token
        self.username = username
        self.base_url = "https://api.twitter.com/2"
    
    def _bearer_oauth(self, r):
        """
        Method required by bearer token authentication.
        """
        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = self.username
        return r
    
    def _get_default_params(
            self,
            method: Literal['/tweets/search', '/tweets/retweeted_by'] = None
        ) -> dict:
        """
        Returns default parameters for API requests.
        """
        if method == '/tweets/search':
            return {
                'expansions': "attachments.poll_ids,attachments.media_keys,author_id,edit_history_tweet_ids,entities.mentions.username,geo.place_id,in_reply_to_user_id,referenced_tweets.id,referenced_tweets.id.author_id",
                'media.fields': "duration_ms,height,media_key,preview_image_url,type,url,width,public_metrics,non_public_metrics,alt_text,variants",
                'tweet.fields': "attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets,source,text,withheld",
                'user.fields': "created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,verified_type,withheld",
            }
        elif method == '/tweets/retweeted_by':
            return {
                'expansions': "pinned_tweet_id",
                'tweet.fields': "attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets,source,text,withheld",
                'user.fields': "created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,verified_type,withheld",
            }
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _log_generator(self, response, params) -> dict:
        """APIをいつどのようなパラメータで呼び出したのか必ず記録するための関数.
        Args:
            response (requests.Response): APIのレスポンスオブジェクト
            params (dict): API呼び出し時のパラメータ
        Returns:
            dict: レスポンス内容とパラメータを含む辞書
        """
        return {
            'metadata': {
                'status_code': response.status_code,
                'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'params': params,
            },
            'response': response.json() if response.status_code == 200 else response.text
        }

    def get_tweets_from_query(self, query, **kwargs):
        """queryからツイートを取得する. 
        
        Args:
            query (str): 取得したい検索クエリ
            kwargs: その他のパラメータ
                - max_results: 取得するツイートの最大数（例: "100"）
                - next_token: ページネーション用のトークン
                - start_time: 取得するツイートの開始日時（例: "2023-01-01T00:00:00Z"）
                - end_time: 取得するツイートの終了日時（例: "2023-01-31T23:59:59Z"）
                - tweet.fields: 取得するツイートのフィールド
                - user.fields: 取得するユーザのフィールド
        
        Returns:
            dict: 取得したツイートの情報を含む辞書
        """
        url = f"{self.base_url}/tweets/search/all"
        
        params = self._get_default_params('/tweets/search')
        params['query'] = query
        params.update(kwargs)
        
        response = requests.get(url, auth=self._bearer_oauth, params=params)
        # print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return self._log_generator(response, params)
    
    def get_tweets_from_user_id(self, user_id: str, **kwargs):
        """user_idからツイートを取得する. 取得されるツイートはリツイートを除く.
        
        Args:
            user_id (str): 取得したいユーザ名
            kwargs: その他のパラメータ
                - max_results: 取得するツイートの最大数（例: "100"）
                - next_token: ページネーション用のトークン
                - start_time: 取得するツイートの開始日時（例: "2023-01-01T00:00:00Z"）
                - end_time: 取得するツイートの終了日時（例: "2023-01-31T23:59:59Z"）
                - tweet.fields: 取得するツイートのフィールド
                - user.fields: 取得するユーザのフィールド
        
        Returns:
            dict: 取得したツイートの情報を含む辞書
        """
        return self.get_tweets_from_query(
            f"from:{user_id} -is:retweet", **kwargs
        )

    def get_retweet_user_from_tweet_id(self, tweet_id, **kwargs):
        """tweet_idからリツイートしたユーザを取得する.

        Args:
            tweet_id (str): 取得したいツイートのID
        kwargs: その他のパラメータ
            - max_results: 取得するユーザの最大数（例: "100"）
            - pagination_token: ページネーション用のトークン
        Returns:
            dict: 取得したリツイートユーザの情報を含む辞書
        """
        url = f"{self.base_url}/tweets/{tweet_id}/retweeted_by"
        
        params = self._get_default_params('/tweets/retweeted_by')
        params.update(kwargs)
        
        response = requests.get(url, auth=self._bearer_oauth, params=params)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return self._log_generator(response, params)
from openai import OpenAI
import time

class Embedding:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def _n_time_embed_trial(self, texts: list[str], model: str = "text-embedding-3-small", n: int = 3, sleep_time: int = 10):
        try: 
            res = self.client.embeddings.create(input=texts, model=model)
        except Exception as e:
            if n > 0:
                print(f"[WARN] Embedding failed, retrying {n} more times: {e}")
                time.sleep(sleep_time)
                return self._n_time_embed_trial(texts, model, n - 1)
            elif n == 0:
                print(f"[WARN] Embedding failed after retries")
                raise e
        return res

    def embed(self, texts: list[str],  model: str = "text-embedding-3-small", retry_num: int=6, sleep_time: int=10) -> list[float]:
        texts = [text.replace("\n", " ") for text in texts]
        if len(texts) > 2048:
            # 分割して処理
            embeddings = []
            for i in range(0, len(texts), 2048):
                batch = texts[i:i + 2048]
                res = self._n_time_embed_trial(input=batch, model=model, n=retry_num, sleep_time=sleep_time)
                embeddings.extend([data.embedding for data in res.data])
            return embeddings
        else:
            res = self._n_time_embed_trial(input = texts, model=model, n=retry_num, sleep_time=sleep_time)
            embeddings = [data.embedding for data in res.data]
            return embeddings
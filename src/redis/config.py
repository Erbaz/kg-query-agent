import redis as redis_module
from typing import Optional


class Redis:
    client: Optional[redis_module.Redis] = None

    def connect(self):
        try:
            self.client = redis_module.Redis(host="localhost", port=6379, db=0)

        except Exception as e:
            print(f"Error initializing Redis: {e}")
            raise e


redis = Redis()

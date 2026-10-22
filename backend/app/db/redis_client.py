import json
from typing import Optional, Any
from app.core.config import settings


class RedisClient:
    def __init__(self):
        self.redis = None
        # Redis connection setup (e.g. redis.Redis(host='redis', port=6379))
        # Disabled by default for local filesystem mode; ready to activate for Docker Compose

    def get_session(self, session_id: str) -> Optional[Any]:
        if not self.redis:
            return None
        data = self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else None

    def set_session(self, session_id: str, data: Any, ttl: int = 3600) -> None:
        if self.redis:
            self.redis.setex(f"session:{session_id}", ttl, json.dumps(data))


redis_client = RedisClient()
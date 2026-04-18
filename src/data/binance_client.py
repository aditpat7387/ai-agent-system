import time
import requests


class BinanceClient:
    def __init__(self, base_url: str = "https://api.binance.com", pause_seconds: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.pause_seconds = pause_seconds
        self.session = requests.Session()

    def get_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 1000):
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        time.sleep(self.pause_seconds)
        return response.json()
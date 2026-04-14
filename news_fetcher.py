"""
news_fetcher.py
===============
News article fetcher with multiple source support.
Primary: NewsAPI.org (free tier: 500 req/day)
Fallback: Yahoo Finance RSS, Finviz

Dependencies:
    pip install requests feedparser

Usage:
    fetcher = NewsFetcher(api_key="YOUR_NEWSAPI_KEY")
    articles = fetcher.fetch("AAPL", max_results=20)
"""

import os
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# ARTICLE MODEL
# ──────────────────────────────────────────────────────────────
class Article:
    __slots__ = ["title", "description", "source", "url", "published_at",
                 "ticker", "sentiment_score"]

    def __init__(self, title: str, description: str, source: str,
                 url: str, published_at: str, ticker: str = ""):
        self.title = (title or "").strip()
        self.description = (description or "").strip()
        self.source = source
        self.url = url
        self.published_at = published_at
        self.ticker = ticker
        self.sentiment_score = None  # filled by engine

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.description}"

    def __repr__(self):
        return f"<Article [{self.source}] {self.title[:50]}...>"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "source": {"name": self.source},
            "url": self.url,
            "publishedAt": self.published_at,
        }


# ──────────────────────────────────────────────────────────────
# SIMPLE REQUEST CACHE (in-memory, TTL-based)
# ──────────────────────────────────────────────────────────────
class RequestCache:
    def __init__(self, ttl_seconds: int = 900):  # 15 min cache
        self._store = {}
        self.ttl = ttl_seconds

    def _key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[list]:
        k = self._key(query)
        if k in self._store:
            data, ts = self._store[k]
            if time.time() - ts < self.ttl:
                return data
        return None

    def set(self, query: str, data: list):
        self._store[self._key(query)] = (data, time.time())

    def clear(self):
        self._store.clear()


# ──────────────────────────────────────────────────────────────
# NEWS FETCHER
# ──────────────────────────────────────────────────────────────
class NewsFetcher:
    """
    Fetches financial news articles from multiple sources.

    Hierarchy:
        1. NewsAPI.org  (best quality, requires API key)
        2. Yahoo Finance RSS  (no key, limited)
        3. Fallback mock data  (for testing / demo)
    """

    NEWSAPI_BASE = "https://newsapi.org/v2/everything"
    YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    RATE_LIMIT_SLEEP = 0.5  # seconds between requests

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY", "")
        self.cache = RequestCache(ttl_seconds=900)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SentimentAlpha/1.0"})

    # ── Primary: NewsAPI ───────────────────────────────────────
    def _fetch_newsapi(self, query: str, max_results: int = 20,
                       days_back: int = 7) -> list[Article]:
        if not self.api_key:
            return []

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "pageSize": min(max_results, 100),
            "sortBy": "publishedAt",
            "from": from_date,
            # Focus on financial domains
            "domains": "reuters.com,bloomberg.com,cnbc.com,wsj.com,ft.com,"
                       "marketwatch.com,barrons.com,thestreet.com,seekingalpha.com",
        }

        try:
            resp = self.session.get(self.NEWSAPI_BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "ok":
                logger.warning(f"NewsAPI error: {data.get('message')}")
                return []

            articles = []
            for item in data.get("articles", []):
                if not item.get("title"):
                    continue
                articles.append(Article(
                    title=item["title"],
                    description=item.get("description", ""),
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    url=item.get("url", ""),
                    published_at=item.get("publishedAt", ""),
                ))
            return articles

        except requests.RequestException as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

    # ── Fallback: Yahoo Finance RSS ────────────────────────────
    def _fetch_yahoo_rss(self, ticker: str, max_results: int = 10) -> list[Article]:
        try:
            import feedparser
        except ImportError:
            return []

        try:
            url = f"{self.YAHOO_RSS}?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_results]:
                articles.append(Article(
                    title=entry.get("title", ""),
                    description=entry.get("summary", ""),
                    source="Yahoo Finance",
                    url=entry.get("link", ""),
                    published_at=entry.get("published", ""),
                    ticker=ticker,
                ))
            return articles
        except Exception as e:
            logger.warning(f"Yahoo RSS failed: {e}")
            return []

    # ── Main fetch method ──────────────────────────────────────
    def fetch(self, ticker: str, max_results: int = 20,
              company_name: Optional[str] = None) -> list[Article]:
        """
        Fetch news articles for a ticker.
        Falls back through sources automatically.
        """
        cache_key = f"{ticker}:{max_results}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit for {ticker}")
            return cached

        query = company_name or ticker
        articles = []

        # Try NewsAPI first
        if self.api_key:
            articles = self._fetch_newsapi(query, max_results)
            time.sleep(self.RATE_LIMIT_SLEEP)

        # Fall back to Yahoo RSS
        if not articles:
            articles = self._fetch_yahoo_rss(ticker, max_results)

        # Mark ticker
        for a in articles:
            a.ticker = ticker

        if articles:
            self.cache.set(cache_key, articles)

        logger.info(f"Fetched {len(articles)} articles for {ticker}")
        return articles

    def fetch_batch(self, tickers: list[str],
                    max_results: int = 20) -> dict[str, list[Article]]:
        """Fetch news for multiple tickers with rate limiting."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.fetch(ticker, max_results)
            time.sleep(self.RATE_LIMIT_SLEEP)
        return results


# ──────────────────────────────────────────────────────────────
# TICKER → COMPANY NAME MAPPER
# ──────────────────────────────────────────────────────────────
TICKER_NAMES = {
    "AAPL": "Apple Inc", "MSFT": "Microsoft", "NVDA": "NVIDIA",
    "GOOGL": "Alphabet Google", "AMZN": "Amazon", "META": "Meta Platforms",
    "TSLA": "Tesla", "JPM": "JPMorgan Chase", "NFLX": "Netflix",
    "AMD": "Advanced Micro Devices", "INTC": "Intel", "CRM": "Salesforce",
    "PLTR": "Palantir", "SNAP": "Snap Inc", "UBER": "Uber",
}

def ticker_to_name(ticker: str) -> str:
    return TICKER_NAMES.get(ticker.upper(), ticker)


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    api_key = os.environ.get("NEWSAPI_KEY", "")
    fetcher = NewsFetcher(api_key=api_key)

    ticker = "AAPL"
    print(f"\nFetching news for {ticker}...")
    articles = fetcher.fetch(ticker, max_results=5,
                              company_name=ticker_to_name(ticker))

    if articles:
        for a in articles:
            print(f"  [{a.source}] {a.title[:70]}")
    else:
        print("  No articles fetched (check API key or network)")
        print("  Set environment variable: export NEWSAPI_KEY=your_key")
        print("  Get free key at: https://newsapi.org")

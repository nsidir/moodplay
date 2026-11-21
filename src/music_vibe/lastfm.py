"""Last.fm tag crawler with a compressed, resumable SQLite cache.

Two passes, both cached so they can be stopped and resumed:
  1. seeds:  tag.getTopTracks for each mood/genre seed tag -> (artist, title, rank)
  2. tracks: track.getTopTags for catalogue tracks, most popular first

Needs LASTFM_API_KEY in .env. The key is never printed or logged.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib

import requests

from .config import LASTFM_API_KEY, LASTFM_DB, LASTFM_USER_AGENT

API = "https://ws.audioscrobbler.com/2.0/"

SEED_TAGS = [
    # moods / feelings
    "chill", "chillout", "relaxing", "calm", "mellow", "peaceful", "soothing", "happy", "uplifting", "feel good",
    "upbeat", "cheerful", "fun", "sad", "melancholic", "melancholy", "depressing", "heartbreak", "emotional",
    "bittersweet", "nostalgic", "dreamy", "atmospheric", "dark", "haunting", "eerie", "angry", "aggressive",
    "energetic", "powerful", "epic", "intense", "romantic", "love", "sexy", "sensual", "lonely", "hopeful",
    "euphoric", "groovy", "funky", "smooth", "soulful", "ethereal", "hypnotic", "trippy", "psychedelic",
    "moody", "brooding", "cinematic", "beautiful", "catchy", "quirky", "weird", "cozy", "warm",
    # activities / settings
    "party", "dance", "workout", "running", "driving", "road trip", "study", "focus", "sleep", "night",
    "late night", "summer", "winter", "autumn", "rainy day", "sunday morning", "morning", "coffee",
    "background", "lounge", "beach", "christmas",
    # sound / style
    "acoustic", "piano", "lo-fi", "ambient", "downtempo", "trip-hop", "chillwave", "shoegaze", "dream pop",
    "post-rock", "minimal", "lounge", "jazz", "soul", "funk", "disco", "house", "techno", "synthwave",
    "indie", "indie pop", "indie folk", "folk", "singer-songwriter", "classical", "orchestral", "metal",
    "punk", "hip-hop", "rap", "r&b", "reggae", "latin", "country", "blues", "electronic", "rock", "pop",
]


class LastFM:
    def __init__(self, rate: float = 4.0):
        if not LASTFM_API_KEY:
            raise SystemExit("LASTFM_API_KEY is not set. Add it to .env (see .env.example).")
        LASTFM_DB.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(LASTFM_DB)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, body BLOB, ts REAL)")
        self.s = requests.Session()
        self.s.headers["User-Agent"] = LASTFM_USER_AGENT
        self.min_gap = 1.0 / rate
        self._last = 0.0
        self._lock = threading.Lock()

    # ---- cache ----
    def cached(self, key: str):
        row = self.db.execute("SELECT body FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def _store(self, key: str, data) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                        (key, zlib.compress(json.dumps(data, separators=(",", ":")).encode(), 9), time.time()))

    # ---- api ----
    def call(self, method: str, key: str, **params):
        hit = self.cached(key)
        if hit is not None:
            return hit
        params.update(method=method, api_key=LASTFM_API_KEY, format="json")
        for attempt in range(5):
            with self._lock:
                wait = self._last + self.min_gap - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                r = self.s.get(API, params=params, timeout=20)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            err = data.get("error")
            if err in (8, 11, 16, 29):  # temporary / rate limit
                time.sleep(2 ** attempt * 3)
                continue
            if err in (10, 26):  # invalid / suspended key: stop, never print the key
                raise SystemExit(f"Last.fm rejected the API key (error {err}): {data.get('message')}")
            self._store(key, data)  # also caches "not found" (error 6) so we don't retry it
            return data
        return {}

    # ---- passes ----
    def seed_pass(self, pages: int = 5, per_page: int = 100) -> None:
        todo = [(t, p) for t in dict.fromkeys(SEED_TAGS) for p in range(1, pages + 1)]
        for n, (tag, page) in enumerate(todo, 1):
            self.call("tag.gettoptracks", f"seed|{tag}|{page}|{per_page}", tag=tag, page=page, limit=per_page)
            if n % 50 == 0:
                self.db.commit()
                print(f"[lastfm] seeds {n}/{len(todo)}")
        self.db.commit()

    def track_pass(self, tracks: list[tuple[str, str, str]], limit: int | None = None) -> None:
        """tracks: (recording_mbid, artist, title), most popular first."""
        tracks = tracks[:limit] if limit else tracks
        t0, done = time.time(), 0
        for n, (mbid, artist, title) in enumerate(tracks, 1):
            key = f"toptags|{mbid}"
            if self.cached(key) is None:
                self.call("track.gettoptags", key, artist=artist, track=title, autocorrect=1)
                done += 1
            if n % 200 == 0:
                self.db.commit()
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(tracks) - n) / rate / 60 if rate else 0
                print(f"[lastfm] tracks {n:,}/{len(tracks):,}  ({rate:.1f}/s, ~{eta:.0f} min left)")
        self.db.commit()


def iter_cache(prefix: str):
    """Yield (key, data) for cached responses whose key starts with prefix (no API key needed)."""
    if not LASTFM_DB.exists():
        return
    db = sqlite3.connect(LASTFM_DB)
    for key, body in db.execute("SELECT key, body FROM cache WHERE key LIKE ?", (prefix + "%",)):
        yield key, json.loads(zlib.decompress(body))

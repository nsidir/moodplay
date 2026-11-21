"""Paths, budgets, and tag filtering."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("MOODPLAY_ROOT") or Path(__file__).resolve().parents[2])
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
DB = DATA / "musicdb"        # the single self-contained dataset (see musicdb.py); env MUSICDB overrides
ARTIFACTS = ROOT / "artifacts"
LASTFM_DB = DATA / "lastfm.sqlite"  # compressed API response cache


def lastfm_zip(split: str) -> Path:
    """lastfm_{train,test}.zip from the project root or data/raw/ (first one found)."""
    for d in (ROOT, RAW):
        if (d / f"lastfm_{split}.zip").exists():
            return d / f"lastfm_{split}.zip"
    return ROOT / f"lastfm_{split}.zip"

LB_SAMPLE_URL = ("https://data.metabrainz.org/pub/musicbrainz/listenbrainz/sample/"
                 "listenbrainz-sample-20250610-094311-full/listenbrainz-sample-dump-20250610-094311.tar.zst")
LB_SAMPLE_SHA256 = "509282544c50b324db8dfb4e33a5af4012f514c153da077a0fb07f093956168b"
LB_SAMPLE = RAW / "lb-sample.tar.zst"
DOWNLOADS = RAW / "downloads"   # extra sources fetched by data/raw/downloads/fetch.sh (make fetch)

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "").strip()
LASTFM_USER_AGENT = "moodplay/0.1 (personal research project)"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TAG_TEMPLATE = "{tag} music"

CATALOG_SIZE = 60_000       # most-listened recordings kept in the catalogue
TOP_TAGS_PER_TRACK = 25
MIN_TAG_TRACKS = 15         # a tag must describe at least this many tracks
SOURCE_WEIGHTS = {          # how much each tag source (musicdb `tags` partition) counts
    "lastfm_api": 1.0,          # Last.fm API crawl: track tags
    "lastfm_seed": 1.0,         # Last.fm API crawl: seed tag charts
    "lastfm_ds": 1.0,           # Last.fm dataset, train split only
    "lastfm_knn": 0.5,          # tags predicted from similar tracks (kNN trained on the train split; artifact)
    "audio": 0.5,               # tags predicted from AcousticBrainz audio features (audio_tags.py; artifact)
    "mb_recording": 1.0, "mb_release_group": 0.6, "mb_artist": 0.35,
}
LASTFM_SOURCES = ("lastfm_api", "lastfm_seed", "lastfm_ds", "lastfm_knn")
API_SOURCES = ("lastfm_api", "lastfm_seed")
PREDICTED_SOURCES = ("lastfm_knn", "audio")  # model outputs: absolute 0-100 confidence, not per-track relative
SIMILARS_PER_TRACK = 30     # strongest Last.fm "similars" kept per track
KNN_TOP_TAGS = 20           # predicted tags kept per track

# tags that carry no vibe: listening habits, ratings, collection notes, chart trivia
_BLOCK_EXACT = {
    "seen live", "favorites", "favourites", "favorite", "favourite", "my favorite", "favorite songs",
    "love", "loved", "awesome", "amazing", "beautiful songs", "best", "good", "great", "cool", "spotify",
    "albums i own", "under 2000 listeners", "fip", "check out", "to listen", "music", "songs",
    "all", "british", "american", "usa", "uk", "english", "german", "french", "female vocalists",
    "male vocalists", "female vocalist", "male vocalist", "singer-songwriter", "cover", "covers",
    "live", "remix", "soundtrack", "instrumental", "seen in concert", "my music", "favourite songs",
    "hits", "top 40", "billboard", "number 1", "grammy", "eurovision", "youtube", "radio",
    "wikidata", "parental advisory", "explicit", "clean", "european", "composer", "film composer",
    "soundtrack composer", "television music", "academy award winner", "pop/rock", "film",
}
_BLOCK_PATTERNS = [
    r"\d+\+? ?(wochen|weeks)", r"^\d+ of \d+ stars", r"^\d+ stars?$", r"^\d{4}$", r"listen",
    r"^favou?rite", r"^my ", r"^best of", r"^albums? ", r"\bseen\b", r"^spotify", r"^\W+$",
    r"http", r"^[a-z]{1,2}$", r"\bcharts?\b", r"award", r"\bcomposer\b",
]
_TAG_FIX = {"hip hop": "hip-hop", "hiphop": "hip-hop", "rnb": "r&b", "r and b": "r&b",
            "electronica": "electronic", "chillout": "chill", "chill out": "chill",
            "trip-hop": "trip hop", "triphop": "trip hop", "lofi": "lo-fi", "lo fi": "lo-fi",
            "melancholy": "melancholic", "singer/songwriter": "singer-songwriter"}


def normalize_tag(tag: str) -> str:
    t = re.sub(r"\s+", " ", tag.strip().lower().replace("_", " "))
    return _TAG_FIX.get(t, t)


def is_vibe_tag(tag: str) -> bool:
    t = normalize_tag(tag)
    if not t or len(t) > 40 or t in _BLOCK_EXACT:
        return False
    return not any(re.search(p, t) for p in _BLOCK_PATTERNS)


# song-identity normalization, shared by Python (Last.fm side) and Spark (catalogue side, build_data.norm_col)
NORM_STEPS = [
    (r"\s*[\(\[][^\)\]]*[\)\]]", ""),                  # (Remastered 2011), [Live]
    (r"\s+-\s+(remaster|live|mono|stereo|single|radio|edit|version).*$", ""),
    (r"\s+(feat\.?|ft\.?|featuring|with)\s.*$", ""),  # feat. X
    (r"[^\w\s]", ""),
    (r"\s+", " "),
]
_NORM_RE = [(re.compile(p), r) for p, r in NORM_STEPS]


def norm_py(s: str) -> str:
    s = (s or "").lower()
    for pat, rep in _NORM_RE:
        s = pat.sub(rep, s)
    return s.strip()


def song_key(artist: str, title: str) -> str:
    return norm_py(artist) + "|" + norm_py(title)

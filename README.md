# music-vibe

Describe a mood, moment or activity in plain language and get music recommendations. Logged-in users rate
tracks 👍 / 👎; ratings personalize results immediately and improve the shared model on retrain.
Sibling project of `movie-vibe-recommender`, built the same way: Spark, sentence embeddings, ALS, Django, Make.

```
make ask Q="chill late night vibes"
make ask Q="I'm stressed" MOOD=match
make run                                 # http://127.0.0.1:8001
```

## Data (all open, < 1 GB on disk)

| Source | Used for | Size |
|---|---|---|
| ListenBrainz sample dump (CC0) | catalogue, MusicBrainz tags/genres, listener counts, 204k loved/hated feedback | 235 MB download, 129 MB parquet |
| Last.fm API (your key) | crowd mood tags: chill, melancholic, study, road trip, ... | cache ~1-3 KB per track |
| Cover Art Archive | album covers (hot-linked) | none |

Spotify, YouTube, Last.fm, MusicBrainz and ListenBrainz links are plain search or page URLs. No unofficial APIs are used.

## Setup

Works on Windows and Linux. Prerequisites: [uv](https://docs.astral.sh/uv/), GNU make and a Java 17 or 21 JDK
(for Spark). The Spark helper finds the JDK through `JAVA_HOME`, the usual install folders
(`C:\Program Files\...`, `/usr/lib/jvm`, sdkman) or `java` on `PATH`.

```
# Debian/Ubuntu
sudo apt install make openjdk-21-jdk
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`data/` and `artifacts/` are gitignored, so a fresh clone needs the steps below (or copy those two folders
from the other machine to skip them). torch is pinned to the CPU-only build on every OS, see `pyproject.toml`.

```
cp .env.example .env          # then put your Last.fm key in .env: LASTFM_API_KEY=...
make install
make extract                  # download + stream the ListenBrainz sample (about 1 min)
make build                    # Spark catalogue + tags (about 3 min) and index (about 1 min)
make crawl TRACKS=20000       # Last.fm tags: seed charts about 3 min, then about 85 min for 20k tracks at 4 req/s
make build                    # rebuild with Last.fm tags
make als                      # Spark implicit ALS (about 2 min)
make run
```

The crawl is cached in `data/lastfm.sqlite` and resumes where it stopped. Ctrl+C is safe.
Use `make crawl TRACKS=5000` for a quicker first pass. After `make extract` you can delete `data/raw/` to free 235 MB.

## How it works

**Spark data step** (`build_data.py`):
- Builds the catalogue from the most-listened ~60k songs.
- Merges duplicate recordings of the same song (normalized artist and title) and drops demos, outtakes and live or instrumental takes.
- Joins tags from four sources: Last.fm (track tags and seed tag charts), MusicBrainz recording, release-group and artist tags.
- Weights each tag as `Σ source_weight · count/max_count`, multiplied by idf, and keeps the top 25 per track.

**Index** (`build_index.py`):
- `E` holds MiniLM embeddings of `"{tag} music"`.
- `M` is the sparse track × tag matrix.
- `V = M·E`, mean-centered and normalized.

**Query** (`moods.py`, `recommend.py`):
- The query is split into aspects, feelings ("I'm stressed" lifts or matches the mood) and exclusions ("no metal").
- A lexicon maps mood and activity words to tags, mixed with embedding matches.
- The score combines tag match, embedding match, popularity and the exclusion penalty.
- An MMR step keeps results diverse, with at most 2 songs per artist.

**ALS** (`als.py`):
- ListenBrainz feedback is about 98% "loved", so the model uses **implicit-feedback ALS** (Spark `implicitPrefs=True`).
- A thumbs-down counts as a confident "no".
- Web users get a personal vector by fold-in, `(YᵀY + Y_Sᵀ(C−I)Y_S + λI)u = Y_SᵀCp`, so personalization starts with the first rating.
- Evaluation: held-out likes hit-rate@50 is 0.20, vs 0.06 for a most-liked baseline.

## Make targets

`make help` lists everything. Common parameters:
- `TRACKS`, `RATE` for the crawl
- `CATALOG`, `TOP_TAGS`, `MIN_TAG` for the data step
- `RANK`, `ITERS`, `REG`, `ALPHA` for ALS
- `K`, `MOOD`, `MIN_USERS`, `YEARS` for queries
- `PORT` for the server (default 8001, so it can run next to the movie app on 8000)

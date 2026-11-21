# moodplay

**Describe how you feel or what you're doing, in your own words, and get music that fits.**

```
make ask Q="chill late night vibes"
make ask Q="I'm stressed" MOOD=match            # match the mood instead of lifting it
make ask Q="rainy sunday, acoustic, no metal" YEARS=1990-
make run                                         # web app on http://127.0.0.1:8001
```

## What it does

- **Understands free text.** "energetic workout, no metal", "something for a road trip" and "I'm sad" all work. A
  query is split into aspects, a feeling ("I'm stressed" lifts you up, or matches the mood with `MOOD=match`) and
  exclusions ("no metal"). A mood/activity lexicon and a sentence-embedding model (MiniLM) turn it into weights over
  ~3,300 vibe tags.
- **Ranks about 46k songs** by how well their tags and tag embeddings match, plus popularity. Results are
  diversified (at most 2 per artist) and come with the tags that explain them, cover art and Spotify, YouTube,
  Last.fm, MusicBrainz and ListenBrainz links.
- **Knows songs' moods even without crowd tags.** Song tags come from MusicBrainz, Last.fm crowd tags and two
  trained models:
  - a kNN model that borrows tags from similar tracks;
  - a neural network that predicts mood tags from audio analysis (AcousticBrainz: danceability, mood classifiers,
    BPM, key, loudness).
- **Learns your taste.** Logged-in users rate songs 👍 / 👎. Ratings personalize results immediately (ALS fold-in) and
  improve the shared implicit-feedback ALS model on retrain.
- **Is measured, not guessed.** The Last.fm dataset's official train/test split is used throughout: models learn only
  from train, and `make evaluate` checks mood queries against held-out test tracks.
- **Runs on one self-contained dataset**, `musicdb` (15.9 GB of open data in Parquet), built by `make fetch db`.

Stack: Python, Spark (data build and ALS), PyTorch (audio model), sentence-transformers, Django (web app), Make.
Works on Linux and Windows.

**Pipeline at a glance**

```
open sources ──make fetch/db──► musicdb (Parquet, versioned schema)
                                  │
             make knn ────────────┤ kNN tags      (Last.fm train split, similar tracks)
             make audio ──────────┤ audio tags    (AcousticBrainz features -> MLP, Last.fm train labels)
             make build ──────────┤ catalogue + weighted tags (Spark) -> tag embeddings -> index
             make als ────────────┤ taste model   (implicit ALS on likes / thumbs)
             make evaluate ───────┘ held-out Last.fm test split
                                  ▼
       query ─► parse + lexicon + MiniLM ─► score 46k songs ─► personalize ─► diversify ─► results
```

## Data: one dataset, `musicdb`

Everything the project reads lives in **`data/musicdb/`**, a single versioned Parquet lake built once from
open sources. After it exists, nothing else is needed: no zips, no dump, no API cache.
Models and indexes are derived from it into `artifacts/`.

| Source (ingested once) | Goes into | Raw size |
|---|---|---|
| ListenBrainz sample dump (CC0) | recordings, artists, release groups, releases, credits, countries, lengths, MusicBrainz tags/genres, popularity (ListenBrainz + MLHD), 204k loved/hated feedback | 235 MB |
| Last.fm dataset (MSD companion): `lastfm_train.zip` + `lastfm_test.zip` | 943k tracks with the official train/test split, 8.6M crowd tags, 56M similar-track pairs | 1.2 GB |
| AcousticBrainz high-level dump (CC0) | `ab_highlevel`: ~30M audio analyses, one row per submission, with 71 classifier probabilities (moods, genres, danceability, timbre, voice) plus file tags | ~40 GB |
| AcousticBrainz low-level features (CC0) | `ab_rhythm`, `ab_lowlevel`, `ab_tonal`: BPM, danceability, onset rate, loudness, dynamics, key/scale for 29.5M submissions | 3 GB |
| Million Song Dataset track list + Echo Nest Taste Profile (research use only) | `msd_tracks` (1M), `plays/source=msd_taste`: 48M user play counts | 0.6 GB |
| ListenBrainz full feedback dump (CC0) | `lb_feedback_events`: 6.5M love/hate events (MessyBrainz-keyed; only ~3.6k carry a recording MBID, so not used by ALS yet) | 82 MB |
| Last.fm API cache (optional, your key) | extra/current crowd tags | ~1-3 KB per track |
| Cover Art Archive | album covers (hot-linked, not stored) | none |

Nothing is thrown away. The large JSON caches keep their full original record in a `raw` column next to the
typed columns, and AcousticBrainz keeps every submission, so new fields can be used later without re-downloading.
Size: 15.9 GB in 21 tables (ab_highlevel alone is ~11 GB).

```
data/musicdb/
  schema.json                    manifest: schema version, every table's columns/types/docs, row counts, provenance
  LICENSES/listenbrainz.txt
  tables/recordings/ artists/ release_groups/ release_group_meta/ releases/ recording_lengths/
         recording_artists/ artist_credits/ artist_countries/ lastfm_tracks/ similars/ msd_tracks/
         lb_feedback_events/ ab_rhythm/ ab_lowlevel/ ab_tonal/
  tables/ab_highlevel/part=<00..29>/
  tables/tags/source=<mb_recording|mb_release_group|mb_artist|lastfm_ds|lastfm_api|lastfm_seed>/
  tables/popularity/source=<listenbrainz|mlhd>/
  tables/feedback/source=<listenbrainz|app>/
  tables/plays/source=<msd_taste>/
```

`make db-info` prints every table and partition with its row count and size. Songs are linked across sources by
`song_key` (normalized `artist|title`), and Last.fm tracks by `track_id` / `tid`.

**Built to stay valid and to grow** (`musicdb.py`):
- The schema contract (`musicdb.CONTRACT`) is the single source of truth. Every write is cast to it and rejects
  unknown or missing required columns. Every read returns exactly the contract schema.
- Adding a table, a nullable column or a source needs no migration: new columns read as null from older files,
  and a new source is a new partition (e.g. web-app thumbs are written to `feedback/source=app` on retrain).
- Breaking changes bump `SCHEMA_VERSION` and ship a migration that runs automatically when the db is opened.
  A db newer than the code is refused instead of misread.
- Writes go to a temp directory and are swapped in with a rename, so a crash never leaves a half-written table.
  Manifest updates take a file lock, so parallel writers (e.g. the AcousticBrainz workers) are safe.
- `make db-check` validates file schemas, row counts, primary keys and references.

Move it between machines as one file: `make db-pack` writes `data/musicdb-v1.tar`, and
`make db-unpack FILE=musicdb-v1.tar` restores it.

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

`data/` and `artifacts/` are gitignored. torch is pinned to the CPU-only build on every OS, see `pyproject.toml`.

```
make install
make fetch                    # extra open sources, ~45 GB download, resumable (about 2.5 h at 5 MB/s)
make db                       # build data/musicdb from every source present (about 1.5 h for ingest)
#   or: make db-unpack FILE=musicdb-v1.tar   (copied from another machine: skips fetch + db)
make knn                      # kNN tag model on the Last.fm train split + test-split eval (about 2 min)
make audio                    # audio -> mood-tag model on AcousticBrainz + test-split eval (first run ~60 min feature extraction, then ~18 min)
make build                    # Spark catalogue + tags (about 3 min) and index (about 2 min)
make als                      # Spark implicit ALS (about 1 min; PLAYS=1 adds Taste Profile, ~10 min)
make evaluate                 # mood queries vs held-out Last.fm test-split tags
make run
```

`make all` runs these steps in order. `make db` only ingests what is not in musicdb yet, so it can be re-run
at any time, e.g. while `make fetch` is still downloading (`uv run moodplay db build --wait` waits for the
remaining AcousticBrainz parts). Raw files are checksum-verified where the publisher provides sums. They are
deleted once ingested, and a `.ingested` marker stops `make fetch` from downloading them again.

The Last.fm dataset is not downloadable from a stable URL: put `lastfm_train.zip` and `lastfm_test.zip` in the
project root or in `data/raw/` before the first `make db`. They are read straight from the zip and never unpacked,
and can be deleted afterwards. The ListenBrainz sample dump is downloaded by `make db` itself.

Optional, on top of the dataset: current tags from the Last.fm API (needs a key). They are ingested into musicdb
as `tags/source=lastfm_api` and `lastfm_seed`:

```
cp .env.example .env          # then put your Last.fm key in .env: LASTFM_API_KEY=...
make crawl TRACKS=20000       # seed charts about 3 min, then about 85 min for 20k tracks at 4 req/s
make build                    # rebuild with the API tags
```

The crawl is cached in `data/lastfm.sqlite` and resumes where it stopped. Ctrl+C is safe.

## Offline training vs online inference

Everything that learns or processes large data runs **offline** in the `make` steps and writes to `artifacts/`.
**Online** (each query), the app only loads those artifacts once and does fast vector arithmetic: no training and no
Spark at query time.

**Offline** (minutes to hours, re-run when data or models change)

| Step | Technique | Engine | Output |
|---|---|---|---|
| `make fetch` / `make db` | streaming ingest into a typed Parquet lake | Python + pyarrow (3 worker processes for AcousticBrainz) | `data/musicdb/` |
| `make knn` | kNN label propagation from similar *train* tracks | SciPy sparse | `knn_tags.parquet` |
| `make audio` | multi-label MLP: audio features → tags, Last.fm *train* labels | PyTorch (CPU) | `audio_tags.parquet` |
| `make build` (data) | dedupe recordings into songs, join 7 tag sources, TF-IDF-like weighting, top 25 tags per song | **Spark** | `tracks` / `track_tags` / `tags.parquet` |
| `make build` (index) | embed every tag with pretrained MiniLM; song vector = weighted mean of its tag embeddings | sentence-transformers, NumPy | `index.npz`, `M.npz` |
| `make als` | implicit-feedback ALS: 48-d taste vectors for songs | **Spark MLlib** | `als_items.npy` |
| `make evaluate` | held-out Last.fm **test** split: ROC-AUC, precision@k, hit-rate@50 | NumPy | `eval_*.json` |

**Online** (well under a second per query; the web app keeps the model in memory and picks up a retrained ALS
file without restarting)

1. **Parse** (rules): feeling ("I'm stressed"), aspects ("chill", "late night"), exclusions ("no metal").
2. **Encode**: MiniLM embeds each aspect. One small transformer forward pass, the only neural network run online.
3. **Query → tag weights**: lexicon tags mixed with the tags whose embeddings are nearest to the query → weight
   vector `tw` over ~3,300 tags and query vector `qv`.
4. **Score all ~46k songs** with two matrix products:
   `0.55·z(M·tw) + 0.45·z(V·qv) + 0.25·z(popularity) − 0.9·exclusion penalty`.
   The kNN and audio models act here through `M`, which already contains their predicted tags.
5. **Personalize** (logged-in users): ALS fold-in solves one 48×48 system from the user's 👍/👎 and the frozen item
   factors, then adds `Y·u` weighted by how many ratings the user has. New ratings take effect immediately.
6. **Diversify**: MMR re-ranks the top 300 against near-duplicates, max 2 songs per artist.
7. **Explain**: top matching tags per song, and "because you liked …" (nearest liked song in ALS space).

**Spark** is used for:
- the catalogue and tag build (`make build`): reading musicdb, collapsing recordings into songs, joining the 7 tag
  sources, tag weighting and top-25 ranking;
- ALS training (`make als`).

It runs in local mode, using all cores of one machine.

## How it works

**Last.fm dataset, train / test** (`knn_tags.py`, `make knn`):
- The official split of the Last.fm dataset: 839k train and 104k test tracks, each with crowd tags (0-100) and
  similar tracks.
- *Train*: train-split tags go into the index (source `lastfm_ds`, matched to the catalogue by `song_key`).
  A kNN tag model is fitted on the train split:
  `tags(t) = Σ s(t,n)·tags(n) / (Σ s(t,n) + 0.5)` over t's 30 most similar tracks n that have train tags.
  It predicts mood tags for every track without train tags (all test tracks and untagged train tracks). These
  enter the index as source `lastfm_knn` (weight 0.5, scaled by the prediction's confidence).
- *Test*: test-split tags never enter the index. They are the held-out ground truth:
  - tag prediction (`artifacts/eval_tags.json`): on 56,644 held-out test tracks, the
    top predicted tag is right 68% of the time (P@1 0.68 vs 0.15 for the most frequent tags; R@10 0.44 vs 0.11).
  - mood queries end to end (`make evaluate`, `artifacts/eval_queries.json`): 58,817 held-out test tracks, represented only by
    what was learned from the train split, are ranked with the production query scoring for 16 probe queries ("sad
    songs", "late night driving", ...). The top 50 are 67% relevant on average vs 2.8% at random (30× lift, mean
    ROC-AUC 0.74). A second check on catalogue songs that only match test tracks is reported too, but only ~70 such
    songs exist (the 2011 dataset overlaps little with today's most-listened songs), so it is usually skipped.

**Audio -> mood tags** (`audio_tags.py`, `make audio`):
- Most catalogue songs have no Last.fm crowd tags (only ~12% match the 2011 dataset), but many have
  AcousticBrainz audio analysis. Features per song: 81 values (71 classifier probabilities such as
  `mood_sad__sad`, `danceability__danceable`, `genre_*`, plus BPM, onset rate, loudness, dynamics, scale), averaged over
  submissions. Songs are linked to Last.fm by `song_key`, taken from the catalogue metadata or from the audio file's own
  artist/title tags, which gives 408k labelled songs.
- *Train*: a multi-label MLP (PyTorch, 81 → 512 → 512 → 439 tags, BCE loss, early stopping) on 324k
  Last.fm **train** songs; 36k train songs are held out for early stopping.
- *Test*: 31.8k **test** songs. Audio alone: P@1 0.44, R@10 0.48, mean mood-tag ROC-AUC 0.80. On the 25.9k test songs
  that also have kNN tags (`artifacts/eval_audio.json`):

  | | P@1 | R@10 | mood-tag AUC |
  |---|---|---|---|
  | kNN (similar tracks) | **0.91** | **0.78** | 0.67 |
  | audio | 0.46 | 0.48 | 0.80 |
  | kNN + audio | 0.86 | 0.75 | **0.84** |

  kNN is best at naming a song's top tags; audio is better at telling which songs are sad, chill, aggressive and so
  on. End to end (`make evaluate`, 43k test tracks with both): mean query AUC goes from 0.737 (kNN) to 0.780
  (kNN + audio), but precision of the top 50 drops from 0.66 to 0.55.
- The audio tags enter the index as source `audio` (weight 0.5) for the 89k catalogue-source recordings with audio.
  Their main value is coverage: they give mood tags to catalogue songs that have no Last.fm tags at all. Tuning the
  `audio` weight in `config.SOURCE_WEIGHTS` with `make evaluate` is the obvious next experiment.

**Spark data step** (`build_data.py`):
- Builds the catalogue from the most-listened ~60k songs.
- Merges duplicate recordings of the same song (normalized artist and title) and drops demos, outtakes and live or instrumental takes.
- Reads only musicdb (plus the kNN artifact). Joins tags from every `tags` partition: Last.fm dataset (train
  tags) and kNN predictions, Last.fm API (optional), and MusicBrainz recording, release-group and artist tags.
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
- Trains on every musicdb `feedback` partition (ListenBrainz plus web-app thumbs, which `make als` syncs in first).
  With `PLAYS=1` it also uses musicdb `plays` (Taste Profile), with confidence growing with log(1 + plays).
- ListenBrainz feedback is about 98% "loved", so the model uses **implicit-feedback ALS** (Spark `implicitPrefs=True`).
- A thumbs-down counts as a confident "no".
- Web users get a personal vector by fold-in, `(YᵀY + Y_Sᵀ(C−I)Y_S + λI)u = Y_SᵀCp`, so personalization starts with the first rating.
- Evaluation: held-out likes hit-rate@50 is 0.20, vs 0.06 for a most-liked baseline.
- Taste Profile play counts (`make als PLAYS=1`) are off by default because they **hurt**: with 5M plays from 600k MSD
  users added, hit-rate@50 on the same held-out ListenBrainz likes fell from 0.199 to 0.091. Those users' tastes (2011,
  MSD-era) swamp the 134k ListenBrainz interactions. Down-weighting them is left for later.

## Make targets

`make help` lists everything. Common parameters:
- `FILE` for `db-pack` / `db-unpack`
- `MIN_TAG` also sets the minimum train-track count for the kNN tag vocabulary in `make knn`
- `TRACKS`, `RATE` for the crawl
- `CATALOG`, `TOP_TAGS`, `MIN_TAG` for the data step
- `RANK`, `ITERS`, `REG`, `ALPHA` for ALS
- `K`, `MOOD`, `MIN_USERS`, `YEARS` for queries
- `PORT` for the server (default 8001, so it can run next to the movie app on 8000)

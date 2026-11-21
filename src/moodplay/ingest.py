"""Build musicdb from the raw sources (run once; afterwards only data/musicdb/ is needed).

  ingest_listenbrainz()  ListenBrainz sample dump (235 MB download, CC0): streamed, never unpacked.
                         Every member is kept: recordings/artists/release-group caches (typed columns +
                         the full JSON record), tags and genres, lengths, credits, countries,
                         release metadata, popularity (ListenBrainz + MLHD) and loved/hated feedback.
  ingest_lastfm()        lastfm_train.zip + lastfm_test.zip (Last.fm dataset, MSD companion): tracks with
                         the official split, all crowd tags and ALL similar-track pairs.
  ingest_lastfm_api()    optional Last.fm API crawl cache (data/lastfm.sqlite) -> tags partitions
                         lastfm_api / lastfm_seed.
  ingest_msd()           Million Song Dataset track list + Echo Nest Taste Profile (48M play counts).
  ingest_lb_feedback_full()  ListenBrainz full feedback dump (6.5M events, MessyBrainz-keyed).
  ingest_ab_features()   AcousticBrainz low-level features (rhythm, loudness, tonal), ~29M submissions each.
  ingest_ab_highlevel()  AcousticBrainz high-level classifiers, 30 dump parts, resumable per part.

Every step is idempotent: it is skipped when its tables are already in musicdb (force=True redoes it).
Raw files of the extra sources live in data/raw/downloads/ (make fetch), are checksum-verified
where the publisher provides sums, and are deleted after a successful ingest unless keep_raw=True.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
import zstandard

from . import musicdb as db
from .config import DOWNLOADS, LB_SAMPLE, LB_SAMPLE_SHA256, LB_SAMPLE_URL, RAW, lastfm_zip, song_key

BATCH = 50_000


# ------------------------------------------------------------------ helpers
def _int(v, lo=None, hi=None):
    try:
        x = int(v)
    except (TypeError, ValueError):
        return None
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        return None
    return x


def _year(v):
    return _int(str(v)[:4] if isinstance(v, str) else v, 1, 9999)


def _json_col(col: pa.ChunkedArray) -> list:
    return [None if v is None else json.dumps(v, ensure_ascii=False) for v in col.to_pylist()]


def _jsonl(fileobj, parse, writer: db.Writer) -> int:
    rows = []
    for line in fileobj:
        rows.append(parse(json.loads(line), line.decode("utf-8").rstrip("\n")))
        if len(rows) >= BATCH:
            writer.write(pd.DataFrame(rows))
            rows.clear()
    if rows:
        writer.write(pd.DataFrame(rows))
    return writer.rows


# ------------------------------------------------------------------ ListenBrainz
def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if LB_SAMPLE.exists() and _sha256(LB_SAMPLE) == LB_SAMPLE_SHA256:
        print("[ingest] ListenBrainz sample dump already downloaded")
        return
    print("[ingest] downloading ListenBrainz sample dump (235 MB) ...")
    tmp = LB_SAMPLE.with_suffix(".part")
    with requests.get(LB_SAMPLE_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if _sha256(tmp) != LB_SAMPLE_SHA256:
        tmp.unlink()
        raise SystemExit("checksum mismatch - download corrupted, try again")
    tmp.replace(LB_SAMPLE)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _recording(d, raw):
    rel, art = d.get("release_data") or {}, d.get("artist_data") or {}
    title = (d.get("recording_data") or {}).get("name")
    artist = art.get("name")
    return {"recording_mbid": d["recording_mbid"], "title": title, "artist": artist,
            "artist_mbids": d.get("artist_mbids") or [], "artist_credit_id": _int(art.get("artist_credit_id")),
            "release_mbid": d.get("release_mbid"), "release": rel.get("name"), "year": _year(rel.get("year")),
            "release_group_mbid": rel.get("release_group_mbid"), "caa_id": _int(rel.get("caa_id")),
            "caa_release_mbid": rel.get("caa_release_mbid"),
            "song_key": song_key(artist or "", title or "") if title else None,
            "last_updated": d.get("last_updated"), "raw": raw}


def _artist(d, raw):
    a = d.get("artist_data") or {}
    return {"artist_mbid": d["artist_mbid"], "name": a.get("name"), "type": a.get("type"),
            "gender": a.get("gender"), "area": a.get("area"), "begin_year": _year(a.get("begin_year")),
            "end_year": _year(a.get("end_year")), "last_updated": d.get("last_updated"), "raw": raw}


def _release_group(d, raw):
    g = d.get("release_group_data") or {}
    return {"release_group_mbid": d["release_group_mbid"], "title": g.get("name"),
            "artist_credit_name": (d.get("artist_data") or {}).get("name"),
            "artist_mbids": d.get("artist_mbids") or [], "type": g.get("type"), "date": g.get("date"),
            "year": _year(g.get("date")), "caa_id": _int(g.get("caa_id")), "caa_release_mbid": g.get("caa_release_mbid"),
            "last_updated": d.get("last_updated"), "raw": raw}


JSONL = {"recordings_cache.jsonl": ("recordings", _recording),
         "artists_cache.jsonl": ("artists", _artist),
         "release_groups_cache.jsonl": ("release_groups", _release_group)}
TAG_FILES = {  # member -> (source partition, kind); all keyed by recording_mbid in the sample
    "recording_tag.parquet": ("mb_recording", "tag"), "recording_genre.parquet": ("mb_recording", "genre"),
    "release_group_tag.parquet": ("mb_release_group", "tag"),
    "release_group_genre.parquet": ("mb_release_group", "genre"),
    "artist_tag.parquet": ("mb_artist", "tag"), "artist_genre.parquet": ("mb_artist", "genre"),
}
POP_FILES = {  # member -> (source, entity_type)
    "top_recording.csv": ("listenbrainz", "recording"), "recording.csv": ("listenbrainz", "recording"),
    "release_group.csv": ("listenbrainz", "release_group"),
    "mlhd_top_recording.csv": ("mlhd", "recording"), "mlhd_recording.csv": ("mlhd", "recording"),
    "mlhd_release_group.csv": ("mlhd", "release_group"),
}


def _plain_parquet(name: str, t: pa.Table):
    """Small parquet members -> their own tables (nested 'artists' kept as JSON)."""
    if name == "recording_length.parquet":
        return "recording_lengths", t.select(["recording_mbid", "length", "recording_id", "is_redirect"])
    if name == "recording_artist.parquet":
        return "recording_artists", pa.table({"recording_mbid": t["recording_mbid"], "artist_mbids": t["artist_mbids"],
                                              "artists_json": _json_col(t["artists"])})
    if name == "artist_credit.parquet":
        return "artist_credits", t
    if name == "artist_country_code.parquet":
        return "artist_countries", t
    if name == "release_metadata.parquet":
        return "releases", pa.table({
            "release_mbid": t["release_mbid"], "release_group_mbid": t["release_group_mbid"],
            "year": [_year(v) for v in t["first_release_date_year"].to_pylist()], "name": t["release_name"],
            "album_artist_name": t["album_artist_name"], "caa_id": t["caa_id"],
            "caa_release_mbid": t["caa_release_mbid"], "artist_mbids": t["artist_credit_mbids"],
            "artists_json": _json_col(t["artists"])})
    if name == "release_group_metadata.parquet":
        return "release_group_meta", pa.table({
            "release_group_mbid": t["release_group_mbid"], "title": t["title"],
            "artist_credit_name": t["artist_credit_name"], "caa_id": t["caa_id"],
            "caa_release_mbid": t["caa_release_mbid"], "artist_mbids": t["artist_credit_mbids"],
            "artists_json": _json_col(t["artists"]),
            "year": [_year(v) for v in t["first_release_date_year"].to_pylist()]})
    return None, None


def ingest_listenbrainz() -> None:
    t0 = time.time()
    download()
    origin = LB_SAMPLE_URL.rsplit("/", 1)[-1]
    tag_w: dict[str, db.Writer] = {}
    pop_w: dict[str, db.Writer] = {}
    try:
        with open(LB_SAMPLE, "rb") as fh:
            stream = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for m in tar:
                    name = m.name.rsplit("/", 1)[-1]
                    if not m.isfile():
                        continue
                    f = tar.extractfile(m)
                    if name in JSONL:
                        table, parse = JSONL[name]
                        with db.Writer(table, origin=f"{origin}:{name}") as w:
                            _jsonl(f, parse, w)
                        print(f"[ingest] {table}: {w.rows:,}  ({time.time() - t0:.0f}s)")
                    elif name in TAG_FILES:
                        src, kind = TAG_FILES[name]
                        t = pq.read_table(io.BytesIO(f.read()))
                        t = pa.table({"item_id": t.column(0), "tag": t.column(1),
                                      "count": t.column(2).cast(pa.float32()), "kind": [kind] * t.num_rows})
                        t = t.filter(pc.is_valid(t["tag"]))
                        if src not in tag_w:
                            tag_w[src] = db.Writer("tags", src, origin)
                        tag_w[src].write(t)
                        print(f"[ingest] tags/{src} += {t.num_rows:,} {kind}s")
                    elif name in POP_FILES:
                        src, ent = POP_FILES[name]
                        df = pd.read_csv(io.BytesIO(f.read()))
                        if len(df):
                            idcol = "recording_mbid" if ent == "recording" else "release_group_mbid"
                            df = pd.DataFrame({"entity_type": ent, "entity_id": df[idcol],
                                               "artist_mbid": df.get("artist_mbid"),
                                               "listens": df.total_listen_count, "users": df.total_user_count})
                            if src not in pop_w:
                                pop_w[src] = db.Writer("popularity", src, origin)
                            pop_w[src].write(df)
                            print(f"[ingest] popularity/{src} += {len(df):,} {ent} rows from {name}")
                    elif name == "recording_feedback.parquet":
                        t = pq.read_table(io.BytesIO(f.read()))
                        n = db.write("feedback", pa.table({"user_id": t["user_id"], "recording_mbid": t["recording_mbid"],
                                                           "value": t["feedback"]}), "listenbrainz", origin)
                        print(f"[ingest] feedback/listenbrainz: {n:,}")
                    elif name.endswith(".parquet"):
                        table, t = _plain_parquet(name, pq.read_table(io.BytesIO(f.read())))
                        if table:
                            print(f"[ingest] {table}: {db.write(table, t, origin=f'{origin}:{name}'):,}")
                        else:
                            print(f"[ingest] skipped unknown member {name} (add it to ingest + musicdb.CONTRACT)")
                    elif name == "COPYING":
                        db.add_license("listenbrainz", f.read().decode("utf-8", "replace"))
                    elif name in ("TIMESTAMP", "SCHEMA_SEQUENCE"):
                        m_ = db.manifest()
                        m_.setdefault("dumps", {}).setdefault("listenbrainz", {"file": origin})[name.lower()] = \
                            f.read().decode().strip()
                        db._save_manifest(m_)
        for w in list(tag_w.values()) + list(pop_w.values()):
            w.commit()
    except BaseException:
        for w in list(tag_w.values()) + list(pop_w.values()):
            w.abort()
        raise
    print(f"[ingest] ListenBrainz done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ Last.fm dataset
SPLITS = ("train", "test")


def ingest_lastfm() -> None:
    t0 = time.time()
    zips = {s: lastfm_zip(s) for s in SPLITS}
    missing = [str(p) for p in zips.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Last.fm dataset not found: {', '.join(missing)} "
                         "(put lastfm_train.zip / lastfm_test.zip in the project root or data/raw/)")
    handles = {s: zipfile.ZipFile(p) for s, p in zips.items()}
    members = {s: [(i, i.filename.rsplit("/", 1)[-1][:-5]) for i in z.infolist() if i.filename.endswith(".json")]
               for s, z in handles.items()}
    # dense integer ids from the file names first, so similars can reference tracks in either split
    tid_of = {tr: n for n, tr in enumerate(tr for s in SPLITS for _, tr in members[s])}
    print(f"[ingest] Last.fm dataset: {', '.join(f'{s} {len(m):,}' for s, m in members.items())} tracks")
    origin = " + ".join(p.name for p in zips.values())
    tw = db.Writer("tags", "lastfm_ds", origin)
    sw = db.Writer("similars", origin=origin)
    kw = db.Writer("lastfm_tracks", origin=origin)
    try:
        tr_rows, tag_rows, sim = [], ([], [], []), ([], [], [], [])

        def flush():
            if tr_rows:
                kw.write(pd.DataFrame(tr_rows))
                tr_rows.clear()
            if tag_rows[0]:
                tw.write(pa.table({"item_id": tag_rows[0], "tag": tag_rows[1],
                                   "count": pa.array(tag_rows[2], pa.float32()), "kind": ["tag"] * len(tag_rows[0])}))
                for c in tag_rows:
                    c.clear()
            if sim[0]:
                sw.write(pa.table({"tid": pa.array(sim[0], pa.int32()), "sim_tid": pa.array(sim[1], pa.int32()),
                                   "score": pa.array(sim[2], pa.float32()), "rank": pa.array(sim[3], pa.int16())}))
                for c in sim:
                    c.clear()

        for split in SPLITS:
            z = handles[split]
            for n, (info, tr) in enumerate(members[split], 1):
                d = json.loads(z.read(info))
                tid = tid_of[tr]
                artist, title = d.get("artist"), d.get("title")
                tr_rows.append({"tid": tid, "track_id": tr, "artist": artist, "title": title, "split": split,
                                "timestamp": d.get("timestamp"), "song_key": song_key(artist or "", title or "")})
                for tag, cnt in d.get("tags") or []:
                    if tag:
                        tag_rows[0].append(tr)
                        tag_rows[1].append(tag)
                        tag_rows[2].append(float(cnt))
                r = 0
                for other, score in d.get("similars") or []:  # strongest first
                    j = tid_of.get(other)
                    if j is None or j == tid:
                        continue
                    sim[0].append(tid)
                    sim[1].append(j)
                    sim[2].append(float(score))
                    sim[3].append(min(r, 32767))
                    r += 1
                if n % BATCH == 0:
                    flush()
                if n % 200_000 == 0:
                    print(f"[ingest] lastfm {split} {n:,}/{len(members[split]):,}  ({time.time() - t0:.0f}s)")
        flush()
        for w in (kw, tw, sw):
            w.commit()
    except BaseException:
        for w in (kw, tw, sw):
            w.abort()
        raise
    finally:
        for z in handles.values():
            z.close()
    print(f"[ingest] lastfm_tracks {kw.rows:,} | tags/lastfm_ds {tw.rows:,} | similars {sw.rows:,} "
          f"({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ Last.fm API cache (optional)
def ingest_lastfm_api() -> None:
    from .config import LASTFM_DB
    from .lastfm import iter_cache

    if not LASTFM_DB.exists():
        print("[ingest] no Last.fm API cache (data/lastfm.sqlite) - skipping (optional, see `make crawl`)")
        return
    api, seed = [], []
    for key, data in iter_cache("toptags|"):
        mbid = key.split("|", 1)[1]
        for t in (data.get("toptags") or {}).get("tag", []) or []:
            if int(t.get("count", 0)) > 0:
                api.append((mbid, t["name"], float(t["count"])))
    for key, data in iter_cache("seed|"):
        _, tag, page, per = key.split("|")
        for i, tr in enumerate((data.get("tracks") or {}).get("track", []) or []):
            rank = (int(page) - 1) * int(per) + i
            score = max(100.0 * (1 - rank / 600), 20.0)  # top of the tag chart counts most
            seed.append((song_key(tr["artist"]["name"], tr["name"]), tag, score))
    for src, rows in (("lastfm_api", api), ("lastfm_seed", seed)):
        df = pd.DataFrame(rows, columns=["item_id", "tag", "count"]).assign(kind="tag")
        print(f"[ingest] tags/{src}: {db.write('tags', df, src, 'data/lastfm.sqlite'):,}")


# ------------------------------------------------------------------ extra sources (data/raw/downloads)
def _ready(name: str) -> Path | None:
    f = DOWNLOADS / name
    return f if f.exists() and (DOWNLOADS / f"{name}.done").exists() else None


def _verify(f: Path, sums_file: str) -> None:
    sums = DOWNLOADS / sums_file
    if not sums.exists():
        return
    want = {ln.split()[1]: ln.split()[0] for ln in sums.read_text().splitlines() if ln.strip()}
    if f.name in want and _sha256(f) != want[f.name]:
        (DOWNLOADS / f"{f.name}.done").unlink(missing_ok=True)
        f.unlink()
        raise SystemExit(f"checksum mismatch for {f.name}: deleted, re-run `make fetch`")


def _done_with(*files: Path, keep_raw: bool = False) -> None:
    if keep_raw:
        return
    for f in files:
        f.unlink(missing_ok=True)
        (DOWNLOADS / f"{f.name}.done").unlink(missing_ok=True)
        (DOWNLOADS / f"{f.name}.ingested").touch()  # tells scripts/fetch_sources.sh not to download it again


def ingest_msd(keep_raw: bool = False) -> None:
    t0 = time.time()
    tracks, mism, trip = _ready("unique_tracks.txt"), _ready("sid_mismatches.txt"), _ready("train_triplets.txt.zip")
    if not (tracks and mism and trip):
        print("[ingest] MSD Taste Profile files not downloaded yet - skipping (make fetch)")
        return
    bad = set()
    for ln in mism.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("ERROR: <"):
            so, tr = ln[8:].split(">", 1)[0].split()
            bad.add((so, tr))
    rows = []
    with open(tracks, encoding="utf-8", errors="replace") as f:
        for ln in f:
            tr, so, artist, title = (ln.rstrip("\n").split("<SEP>") + ["", "", ""])[:4]
            rows.append((tr, so, artist, title, song_key(artist, title), (so, tr) in bad))
    df = pd.DataFrame(rows, columns=["track_id", "song_id", "artist", "title", "song_key", "mismatch"])
    print(f"[ingest] msd_tracks: {db.write('msd_tracks', df, origin=tracks.name):,} ({df.mismatch.sum():,} mismatches)")
    with db.Writer("plays", "msd_taste", trip.name) as w, zipfile.ZipFile(trip) as z, \
            z.open("train_triplets.txt") as fh:
        for chunk in pd.read_csv(fh, sep="\t", names=["user_id", "item_id", "plays"], chunksize=5_000_000,
                                 dtype={"user_id": "string", "item_id": "string", "plays": "int32"}):
            w.write(chunk)
            print(f"[ingest] plays/msd_taste {w.rows:,}  ({time.time() - t0:.0f}s)")
    _done_with(tracks, mism, trip, keep_raw=keep_raw)


def _feedback(v) -> int:
    """1 / -1 / 0 as numbers or strings, or 'love' / 'hate'."""
    if isinstance(v, str) and not v.lstrip("-").isdigit():
        return {"love": 1, "loved": 1, "hate": -1, "hated": -1}.get(v.strip().lower(), 0)
    return int(v)


def ingest_lb_feedback_full(keep_raw: bool = False) -> None:
    files = sorted(DOWNLOADS.glob("listenbrainz-feedback-dump-*.tar.zst"))
    f = _ready(files[-1].name) if files else None
    if not f:
        print("[ingest] ListenBrainz full feedback dump not downloaded yet - skipping (make fetch)")
        return
    with db.Writer("lb_feedback_events", origin=f.name) as w, open(f, "rb") as fh:
        stream = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            rows = []
            for m in tar:
                if not (m.isfile() and m.name.endswith("data.json")):
                    continue
                for line in tar.extractfile(m):
                    d = json.loads(line)
                    nz = lambda v: None if v in (None, "None", "") else v  # noqa: E731
                    rows.append({"user_name": d["user_name"], "recording_msid": nz(d.get("recording_msid")),
                                 "recording_mbid": nz(d.get("recording_mbid")), "value": _feedback(d["feedback"]),
                                 "created": d.get("created")})
                if len(rows) >= 500_000:
                    w.write(pd.DataFrame(rows))
                    rows.clear()
            if rows:
                w.write(pd.DataFrame(rows))
    print(f"[ingest] lb_feedback_events: {w.rows:,}")
    _done_with(f, keep_raw=keep_raw)


class _Stream(io.RawIOBase):
    """Non-seekable wrapper so pandas can read a member of a streamed (r|) tar."""

    def __init__(self, f):
        self.f = f

    def readable(self):
        return True

    def readinto(self, b):
        data = self.f.read(len(b))
        b[:len(data)] = data
        return len(data)


def _text(f):
    return io.TextIOWrapper(io.BufferedReader(_Stream(f), 1 << 20), encoding="utf-8", errors="replace")


AB_FEATURES = {"rhythm": "ab_rhythm", "lowlevel": "ab_lowlevel", "tonal": "ab_tonal"}


def ingest_ab_features(keep_raw: bool = False, force: bool = False) -> None:
    for kind, table in AB_FEATURES.items():
        if db.has(table) and not force:
            continue
        f = _ready(f"acousticbrainz-lowlevel-features-20220623-{kind}.tar.zst")
        if not f:
            print(f"[ingest] AcousticBrainz {kind} features not downloaded yet - skipping (make fetch)")
            continue
        t0 = time.time()
        _verify(f, "ab-lowlevel-features.sha256sums")
        cols = db.CONTRACT[table].file_schema
        with db.Writer(table, origin=f.name) as w, open(f, "rb") as fh:
            stream = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for m in tar:
                    if not (m.isfile() and m.name.endswith(".csv")):
                        continue
                    for chunk in pd.read_csv(_text(tar.extractfile(m)), chunksize=2_000_000, low_memory=False):
                        for c in chunk.columns:
                            if c not in ("mbid", "key_key", "key_scale"):
                                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
                        chunk = chunk[[c for c in cols.names if c in chunk.columns]].dropna(subset=["mbid"])
                        w.write(chunk)
                        print(f"[ingest] {table} {w.rows:,}  ({time.time() - t0:.0f}s)")
        _done_with(f, keep_raw=keep_raw)


def _ab_row(d: dict, mbid: str, offset: int) -> dict:
    hl, meta = d.get("highlevel") or {}, d.get("metadata") or {}
    row = {"mbid": mbid, "submission_offset": offset}
    for clf, classes in db.AB_CLASSIFIERS.items():
        probs = (hl.get(clf) or {}).get("all") or {}
        for k in classes:
            row[db.ab_col(clf, k)] = probs.get(k)
    ap, tags = meta.get("audio_properties") or {}, meta.get("tags") or {}

    def tag(k):
        v = tags.get(k)
        return "; ".join(map(str, v)) if isinstance(v, list) else (None if v is None else str(v))

    row.update(length=ap.get("length"), bit_rate=_int(ap.get("bit_rate")), codec=ap.get("codec"),
               lossless=None if ap.get("lossless") is None else bool(ap.get("lossless")),
               sample_rate=_int(ap.get("sample_rate")), replay_gain=ap.get("replay_gain"),
               md5_encoded=ap.get("md5_encoded"), tag_artist=tag("artist"), tag_title=tag("title"),
               tag_album=tag("album"), tag_date=tag("date"), tag_genre=tag("genre"), tag_mood=tag("mood"),
               tag_bpm=tag("bpm"), essentia_version=((meta.get("version") or {}).get("highlevel") or {}).get("essentia"))
    return row


AB_PARTS = 30


def ingest_ab_highlevel(keep_raw: bool = False, wait: bool = False, force: bool = False, workers: int = 3) -> None:
    """One musicdb partition per dump part, so the ingest resumes where it stopped.
    wait=True polls for parts that are still downloading (make fetch running in parallel).
    Parts are spread over `workers` processes (parsing ~1M JSON documents per part is CPU-bound)."""
    todo = [p for p in range(AB_PARTS) if force or not db.has("ab_highlevel", f"{p:02d}")]
    if workers <= 1 or len(todo) <= 1:
        _ab_worker(todo, keep_raw, wait)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_ab_worker, args=(todo[i::workers], keep_raw, wait)) for i in range(workers)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    failed = [pr.exitcode for pr in procs if pr.exitcode]
    if failed:
        raise SystemExit(f"ab_highlevel: {len(failed)} worker(s) failed; re-run to resume the missing parts")


def _ab_worker(todo: list[int], keep_raw: bool, wait: bool) -> None:
    todo = list(todo)
    while todo:
        progressed = False
        for p in list(todo):
            f = _ready(f"acousticbrainz-highlevel-json-20220623-{p}.tar.zst")
            if not f:
                continue
            t0 = time.time()
            _verify(f, "ab-highlevel.sha256sums")
            with db.Writer("ab_highlevel", f"{p:02d}", origin=f.name) as w, open(f, "rb") as fh:
                stream = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    rows = []
                    for m in tar:
                        if not (m.isfile() and m.name.endswith(".json")):
                            continue
                        stem = m.name.rsplit("/", 1)[-1][:-5]
                        mbid, _, off = stem.rpartition("-")
                        rows.append(_ab_row(json.loads(tar.extractfile(m).read()), mbid, int(off)))
                        if len(rows) >= 25_000:
                            w.write(pd.DataFrame(rows))
                            rows.clear()
                    if rows:
                        w.write(pd.DataFrame(rows))
            print(f"[ingest] ab_highlevel part {p:02d}: {w.rows:,} submissions ({time.time() - t0:.0f}s)")
            _done_with(f, keep_raw=keep_raw)
            todo.remove(p)
            progressed = True
        if todo and not progressed:
            if not wait:
                print(f"[ingest] ab_highlevel: {len(todo)} parts not downloaded yet - skipping (make fetch)")
                return
            time.sleep(30)


def run(force: bool = False, wait: bool = False) -> None:
    """Ingest every source whose raw files are present and whose tables are missing (or all with force)."""
    def need(*tables):
        return force or not all(db.has(t) for t in tables)

    if need("recordings", "tags"):
        ingest_listenbrainz()
    if need("lastfm_tracks", "similars"):
        ingest_lastfm()
    ingest_lastfm_api()
    if need("msd_tracks", "plays"):
        ingest_msd()
    if need("lb_feedback_events"):
        ingest_lb_feedback_full()
    ingest_ab_features(force=force)
    ingest_ab_highlevel(force=force, wait=wait)
    db.check()

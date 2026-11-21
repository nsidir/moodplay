"""musicdb: the project's single, self-contained dataset (a versioned Parquet lake).

    data/musicdb/
      schema.json                   manifest: schema version, tables, columns, row counts, provenance
      LICENSES/                     licence texts shipped with the sources
      tables/<table>/part-*.parquet                 plain tables
      tables/<table>/source=<src>/part-*.parquet    tables partitioned by source (tags, popularity, feedback)

Everything the project needs is in here: once it is built (`make db`) or unpacked (`make db-unpack`),
no zip, dump or API cache is needed. Derived outputs (models, indexes) live in artifacts/.

The CONTRACT below is the source of truth. Every write is validated against it, and every read
returns the contract schema, so the data always looks the same to the code that uses it.

Evolution rules (so old databases keep working):
  * add a table or a nullable column -> just add it to CONTRACT; old files read it as null.
  * add a data source               -> write a new partition (tags/popularity/feedback) with a new `source`.
  * anything breaking (rename, type change, drop) -> bump SCHEMA_VERSION and add a function to
    MIGRATIONS that rewrites the affected tables; it runs automatically on the next open.
Writes go to a temp directory and are swapped in with a rename, so a crash never leaves a half-written table.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import DB

SCHEMA_VERSION = 1
FORMAT = "musicdb"
S, I16, I32, I64, F32, B = pa.string(), pa.int16(), pa.int32(), pa.int64(), pa.float32(), pa.bool_()
LS = pa.list_(pa.string())


@dataclass
class Col:
    name: str
    type: pa.DataType
    doc: str
    nullable: bool = True


@dataclass
class Table:
    doc: str
    cols: list[Col]
    pk: list[str] = field(default_factory=list)
    partition: str | None = None   # hive partition column (always a string)
    required: bool = True          # `check` fails if a required table is missing

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([pa.field(c.name, c.type, nullable=c.nullable) for c in self.cols])

    @property
    def file_schema(self) -> pa.Schema:  # columns physically stored in the parquet files
        return pa.schema([f for f in self.schema if f.name != self.partition])


RAW = Col("raw", S, "full original JSON record (lossless; extract more columns from it later)")
UPD = Col("last_updated", S, "source cache timestamp")

CONTRACT: dict[str, Table] = {
    # ---------------- ListenBrainz / MusicBrainz (sample dump, CC0) ----------------
    "recordings": Table("MusicBrainz recordings (ListenBrainz recordings cache)", [
        Col("recording_mbid", S, "MusicBrainz recording id", False),
        Col("title", S, "recording title"),
        Col("artist", S, "artist credit name"),
        Col("artist_mbids", LS, "artist MBIDs of the credit"),
        Col("artist_credit_id", I64, "MusicBrainz artist credit id"),
        Col("release_mbid", S, "release the cache picked for this recording"),
        Col("release", S, "release name"),
        Col("year", I16, "release year"),
        Col("release_group_mbid", S, "release group MBID"),
        Col("caa_id", I64, "Cover Art Archive image id"),
        Col("caa_release_mbid", S, "release MBID that owns the cover"),
        Col("song_key", S, "normalized 'artist|title' (config.song_key), links songs across sources"),
        UPD, RAW], pk=["recording_mbid"]),
    "recording_lengths": Table("recording length in ms", [
        Col("recording_mbid", S, "recording MBID", False), Col("length", I64, "length in milliseconds"),
        Col("recording_id", I64, "MusicBrainz row id"), Col("is_redirect", B, "MBID is a redirect")],
        pk=["recording_mbid"]),
    "recording_artists": Table("artists credited on each recording", [
        Col("recording_mbid", S, "recording MBID", False), Col("artist_mbids", LS, "artist MBIDs"),
        Col("artists_json", S, "credit details as JSON (name, mbid, join phrase)")], pk=["recording_mbid"]),
    "artists": Table("MusicBrainz artists (ListenBrainz artists cache)", [
        Col("artist_mbid", S, "artist MBID", False), Col("name", S, "artist name"),
        Col("type", S, "Person, Group, ..."), Col("gender", S, "gender if a person"),
        Col("area", S, "area name"), Col("begin_year", I16, "begin year"), Col("end_year", I16, "end year"),
        UPD, RAW], pk=["artist_mbid"]),
    "artist_countries": Table("artist country codes", [
        Col("artist_mbid", S, "artist MBID", False), Col("artist_name", S, "name"),
        Col("country_code", S, "ISO 3166 alpha-3", False)], pk=["artist_mbid", "country_code"]),
    "artist_credits": Table("artist credit -> artists", [
        Col("artist_credit_id", I64, "artist credit id", False), Col("artist_mbid", S, "artist MBID", False),
        Col("position", I32, "position in the credit"), Col("join_phrase", S, "join phrase")],
        pk=["artist_credit_id", "position"]),
    "release_groups": Table("MusicBrainz release groups (ListenBrainz release groups cache + metadata)", [
        Col("release_group_mbid", S, "release group MBID", False), Col("title", S, "title"),
        Col("artist_credit_name", S, "artist credit"), Col("artist_mbids", LS, "artist MBIDs"),
        Col("type", S, "Album, Single, EP, ..."), Col("date", S, "first release date"),
        Col("year", I16, "first release year"), Col("caa_id", I64, "cover id"),
        Col("caa_release_mbid", S, "release owning the cover"), UPD, RAW], pk=["release_group_mbid"]),
    "release_group_meta": Table("release group metadata shipped as a separate file in the dump", [
        Col("release_group_mbid", S, "release group MBID", False), Col("title", S, "title"),
        Col("artist_credit_name", S, "artist credit"), Col("caa_id", I64, "cover id"),
        Col("caa_release_mbid", S, "release owning the cover"), Col("artist_mbids", LS, "artist MBIDs"),
        Col("artists_json", S, "credit details as JSON"), Col("year", I16, "first release year")],
        pk=["release_group_mbid"]),
    "releases": Table("MusicBrainz releases", [
        Col("release_mbid", S, "release MBID", False), Col("release_group_mbid", S, "release group"),
        Col("year", I16, "first release year"), Col("name", S, "release name"),
        Col("album_artist_name", S, "album artist"), Col("caa_id", I64, "cover id"),
        Col("caa_release_mbid", S, "release owning the cover"), Col("artist_mbids", LS, "artist MBIDs"),
        Col("artists_json", S, "credit details as JSON")], pk=["release_mbid"]),
    "popularity": Table("listen counts per entity and source", [
        Col("entity_type", S, "'recording' or 'release_group'", False),
        Col("entity_id", S, "MBID of the entity", False),
        Col("artist_mbid", S, "artist the count is attributed to (recording rows)"),
        Col("listens", I64, "total listens"), Col("users", I64, "distinct listeners"),
        Col("source", S, "listenbrainz | mlhd | ...", False)], partition="source"),
    "feedback": Table("user feedback events on recordings: +1 love / -1 hate (an event log: the source repeats "
                      "rows, so there is no unique key; consumers keep the last value per user and recording)", [
        Col("user_id", I64, "user id, unique within its source", False),
        Col("recording_mbid", S, "recording MBID", False), Col("value", pa.int8(), "+1 or -1", False),
        Col("created", S, "timestamp if known"),
        Col("source", S, "listenbrainz | app | ...", False)], partition="source"),
    # ---------------- tags from every source ----------------
    "tags": Table("tags / genres with counts; item_id meaning depends on source (see manifest `sources`)", [
        Col("item_id", S, "recording MBID, Last.fm track_id or song_key (per source)", False),
        Col("tag", S, "tag as given by the source (normalize with config.normalize_tag)", False),
        Col("count", F32, "source-specific strength (MusicBrainz votes, Last.fm 0-100, chart score)", False),
        Col("kind", S, "'tag' or 'genre'"),
        Col("source", S, "mb_recording | mb_release_group | mb_artist | lastfm_ds | lastfm_api | lastfm_seed",
            False)], partition="source"),
    # ---------------- Last.fm dataset (Million Song Dataset companion) ----------------
    "lastfm_tracks": Table("Last.fm dataset tracks with the official train/test split", [
        Col("tid", I32, "dense integer id (0..n-1), used by `similars`", False),
        Col("track_id", S, "MSD track id (TR...)", False), Col("artist", S, "artist"), Col("title", S, "title"),
        Col("split", S, "'train' or 'test' (official split)", False),
        Col("timestamp", S, "Last.fm crawl time"),
        Col("song_key", S, "normalized 'artist|title' (links to recordings.song_key)")], pk=["tid"]),
    "similars": Table("Last.fm track-to-track similarity (all pairs from the dataset)", [
        Col("tid", I32, "lastfm_tracks.tid", False), Col("sim_tid", I32, "similar lastfm_tracks.tid", False),
        Col("score", F32, "similarity (0-1]", False),
        Col("rank", I16, "0 = most similar neighbour of tid", False)], pk=["tid", "sim_tid"]),
    # ---------------- Million Song Dataset: track list + Echo Nest Taste Profile ----------------
    "msd_tracks": Table("Million Song Dataset tracks (unique_tracks.txt): track id <-> song id", [
        Col("track_id", S, "MSD track id (TR...), = lastfm_tracks.track_id", False),
        Col("song_id", S, "MSD song id (SO...), used by plays/source=msd_taste", False),
        Col("artist", S, "artist"), Col("title", S, "title"),
        Col("song_key", S, "normalized 'artist|title'"),
        Col("mismatch", B, "listed in sid_mismatches.txt: song<->track match known to be wrong", False)],
        pk=["track_id"], required=False),
    "plays": Table("implicit listening counts (user x item)", [
        Col("user_id", S, "user id, unique within its source", False),
        Col("item_id", S, "item id; msd_taste: MSD song id (msd_tracks.song_id)", False),
        Col("plays", I32, "play count", False),
        Col("source", S, "msd_taste | ...", False)], pk=["source", "user_id", "item_id"], partition="source",
        required=False),
    "lb_feedback_events": Table("ListenBrainz full feedback dump (all users, MessyBrainz-keyed; few rows carry an MBID)", [
        Col("user_name", S, "ListenBrainz user name", False), Col("recording_msid", S, "MessyBrainz id"),
        Col("recording_mbid", S, "MusicBrainz recording id when present"),
        Col("value", pa.int8(), "+1 love / -1 hate", False), Col("created", S, "timestamp")], required=False),
    # ---------------- AcousticBrainz (CC0): audio analysis per recording submission ----------------
    "ab_highlevel": Table("AcousticBrainz high-level classifiers: one row per submission, columns "
                          "<classifier>__<class> = probability", [
        Col("mbid", S, "recording MBID", False), Col("submission_offset", I32, "n-th submission of the MBID", False),
        # <classifier>__<class> probability columns are inserted here from AB_CLASSIFIERS (below)
        Col("length", F32, "audio length (s)"), Col("bit_rate", I32, "bit rate"), Col("codec", S, "codec"),
        Col("lossless", B, "lossless source"), Col("sample_rate", I32, "sample rate"),
        Col("replay_gain", F32, "replay gain (dB)"), Col("md5_encoded", S, "md5 of the encoded audio"),
        Col("tag_artist", S, "file tag"), Col("tag_title", S, "file tag"), Col("tag_album", S, "file tag"),
        Col("tag_date", S, "file tag"), Col("tag_genre", S, "file tag(s), '; '-joined"),
        Col("tag_mood", S, "file tag(s), '; '-joined"), Col("tag_bpm", S, "file tag"),
        Col("essentia_version", S, "high-level extractor version"),
        Col("part", S, "dump part 00-29", False)], partition="part", required=False),
    "ab_rhythm": Table("AcousticBrainz low-level rhythm features per submission", [
        Col("mbid", S, "recording MBID", False), Col("submission_offset", I32, "submission", False),
        Col("bpm", F32, "tempo"), Col("bpm_histogram_first_peak_bpm_mean", F32, ""),
        Col("bpm_histogram_first_peak_bpm_median", F32, ""), Col("bpm_histogram_second_peak_bpm_mean", F32, ""),
        Col("bpm_histogram_second_peak_bpm_median", F32, ""), Col("danceability", F32, "0-3 DFA danceability"),
        Col("onset_rate", F32, "onsets per second")], pk=["mbid", "submission_offset"], required=False),
    "ab_lowlevel": Table("AcousticBrainz low-level loudness features per submission", [
        Col("mbid", S, "recording MBID", False), Col("submission_offset", I32, "submission", False),
        Col("average_loudness", F32, "0-1"), Col("dynamic_complexity", F32, "dB"),
        Col("mfcc_zero_mean", F32, "MFCC 0 mean (energy)")], pk=["mbid", "submission_offset"], required=False),
    "ab_tonal": Table("AcousticBrainz low-level tonal features per submission", [
        Col("mbid", S, "recording MBID", False), Col("submission_offset", I32, "submission", False),
        Col("key_key", S, "key"), Col("key_scale", S, "major/minor"), Col("tuning_frequency", F32, "Hz"),
        Col("tuning_equal_tempered_deviation", F32, "")], pk=["mbid", "submission_offset"], required=False),
}

AB_CLASSIFIERS = {
    "danceability": ["danceable", "not_danceable"], "gender": ["female", "male"],
    "genre_dortmund": ["alternative", "blues", "electronic", "folkcountry", "funksoulrnb", "jazz", "pop",
                       "raphiphop", "rock"],
    "genre_electronic": ["ambient", "dnb", "house", "techno", "trance"],
    "genre_rosamerica": ["cla", "dan", "hip", "jaz", "pop", "rhy", "roc", "spe"],
    "genre_tzanetakis": ["blu", "cla", "cou", "dis", "hip", "jaz", "met", "pop", "reg", "roc"],
    "ismir04_rhythm": ["ChaChaCha", "Jive", "Quickstep", "Rumba-American", "Rumba-International", "Rumba-Misc",
                       "Samba", "Tango", "VienneseWaltz", "Waltz"],
    "mood_acoustic": ["acoustic", "not_acoustic"], "mood_aggressive": ["aggressive", "not_aggressive"],
    "mood_electronic": ["electronic", "not_electronic"], "mood_happy": ["happy", "not_happy"],
    "mood_party": ["not_party", "party"], "mood_relaxed": ["not_relaxed", "relaxed"], "mood_sad": ["not_sad", "sad"],
    "moods_mirex": ["Cluster1", "Cluster2", "Cluster3", "Cluster4", "Cluster5"],
    "timbre": ["bright", "dark"], "tonal_atonal": ["atonal", "tonal"], "voice_instrumental": ["instrumental", "voice"],
}


def ab_col(classifier: str, cls: str) -> str:
    return f"{classifier}__{cls}".lower().replace("-", "_")


CONTRACT["ab_highlevel"].cols[2:2] = [Col(ab_col(c, k), F32, f"{c}: P({k})")
                                      for c, ks in AB_CLASSIFIERS.items() for k in ks]

SOURCES = {  # provenance shown in the manifest; item_id meaning for tags
    "mb_recording": "tags: item_id = recording MBID; MusicBrainz recording tags/genres (ListenBrainz sample, CC0)",
    "mb_release_group": "tags: item_id = recording MBID; tags/genres of the recording's release group",
    "mb_artist": "tags: item_id = recording MBID; tags/genres of the recording's artists",
    "lastfm_ds": "tags: item_id = lastfm_tracks.track_id; Last.fm dataset crowd tags (both splits; "
                 "filter with lastfm_tracks.split, test tags are for evaluation only)",
    "lastfm_api": "tags: item_id = recording MBID; Last.fm API track.getTopTags (optional crawl)",
    "lastfm_seed": "tags: item_id = song_key; Last.fm API tag.getTopTracks chart position score (optional crawl)",
    "listenbrainz": "popularity / feedback from the ListenBrainz sample dump (CC0)",
    "mlhd": "popularity from the Music Listening Histories Dataset (via ListenBrainz)",
    "app": "feedback: thumbs up/down from the web app (user_id = Django user id)",
    "msd_taste": "plays: Echo Nest Taste Profile (MSD challenge train_triplets; research use only)",
}


PK_CHECK_MAX_ROWS = 20_000_000


class SchemaError(RuntimeError):
    pass


# ------------------------------------------------------------------ manifest
def root() -> Path:
    return Path(os.environ.get("MUSICDB", DB))


def _manifest_path() -> Path:
    return root() / "schema.json"


def manifest() -> dict:
    p = _manifest_path()
    if not p.exists():
        return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "created_at": _now(), "tables": {},
                "sources": {}}
    return json.loads(p.read_text())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _save_manifest(m: dict) -> None:
    root().mkdir(parents=True, exist_ok=True)
    m["format"], m["updated_at"] = FORMAT, _now()
    m["contract"] = {name: {"doc": t.doc, "primary_key": t.pk, "partition_by": t.partition,
                            "columns": [{"name": c.name, "type": str(c.type), "nullable": c.nullable, "doc": c.doc}
                                        for c in t.cols]} for name, t in CONTRACT.items()}
    m["evolution"] = ("additive changes (new tables, new nullable columns, new partitions) need no migration; "
                      "breaking changes bump schema_version and ship a migration in musicdb.MIGRATIONS")
    tmp = _manifest_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(_manifest_path())


@contextlib.contextmanager
def _locked():
    """Exclusive lock around manifest read-modify-write, so parallel writers (other processes) don't lose updates."""
    root().mkdir(parents=True, exist_ok=True)
    with open(root() / ".lock", "a+b") as fh:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def exists() -> bool:
    return _manifest_path().exists()


def open_db() -> dict:
    """Validate the version and run pending migrations. Call before reading."""
    if not exists():
        raise SystemExit(f"No musicdb at {root()}. Build it with `make db` or unpack one with `make db-unpack`.")
    m = manifest()
    v = m.get("schema_version", 0)
    if v > SCHEMA_VERSION:
        raise SchemaError(f"musicdb schema v{v} is newer than this code (v{SCHEMA_VERSION}); update the code.")
    while v < SCHEMA_VERSION:
        print(f"[musicdb] migrating schema v{v} -> v{v + 1}")
        MIGRATIONS[v]()
        m = manifest()
        v = m["schema_version"] = v + 1
        _save_manifest(m)
    return m


MIGRATIONS: dict[int, callable] = {}  # {from_version: fn} - fn rewrites tables in place via write()


# ------------------------------------------------------------------ paths / reading
def table_dir(name: str) -> Path:
    if name not in CONTRACT:
        raise KeyError(f"unknown table {name!r}; tables: {', '.join(CONTRACT)}")
    return root() / "tables" / name


def path(name: str) -> str:
    """Directory for Spark: spark.read.option('mergeSchema', 'true').parquet(musicdb.path('tags'))."""
    return str(table_dir(name))


def has(name: str, partition: str | None = None) -> bool:
    d = table_dir(name)
    if partition is not None:
        d = d / f"{CONTRACT[name].partition}={partition}"
    return d.exists() and any(d.rglob("*.parquet"))


def partitions(name: str) -> list[str]:
    t = CONTRACT[name]
    d = table_dir(name)
    return sorted(p.name.split("=", 1)[1] for p in d.glob(f"{t.partition}=*")) if t.partition and d.exists() else []


def dataset(name: str) -> ds.Dataset:
    t = CONTRACT[name]
    d = table_dir(name)
    if not has(name):
        raise SystemExit(f"musicdb table {name!r} is empty or missing ({d}). Run `make db`.")
    part = ds.partitioning(pa.schema([(t.partition, pa.string())]), flavor="hive") if t.partition else None
    # the contract schema: columns missing in older files come back as nulls
    return ds.dataset(str(d), format="parquet", partitioning=part, schema=t.schema)


def read_arrow(name: str, columns: list[str] | None = None, filter=None) -> pa.Table:
    return dataset(name).to_table(columns=columns, filter=filter)


def read(name: str, columns: list[str] | None = None, filter=None, sources: list[str] | None = None) -> pd.DataFrame:
    """Read a table as pandas. `filter` is a pyarrow expression (pyarrow.dataset.field('x') == 1)."""
    t = CONTRACT[name]
    if sources is not None:
        f = ds.field(t.partition).isin(sources)
        filter = f if filter is None else filter & f
    return read_arrow(name, columns, filter).to_pandas()


# ------------------------------------------------------------------ writing
def conform(name: str, data, partition: str | None = None) -> pa.Table:
    """Cast data (pandas or arrow) to the table's file schema; fill missing nullable columns with nulls."""
    t = CONTRACT[name]
    tbl = data if isinstance(data, pa.Table) else pa.Table.from_pandas(data, preserve_index=False)
    if t.partition and t.partition in tbl.column_names:
        vals = pc.unique(tbl[t.partition]).to_pylist()
        if vals != [partition]:
            raise SchemaError(f"{name}: column {t.partition} {vals} does not match partition {partition!r}")
        tbl = tbl.drop([t.partition])
    extra = set(tbl.column_names) - set(t.file_schema.names)
    if extra:
        raise SchemaError(f"{name}: columns {sorted(extra)} are not in the contract; add them to musicdb.CONTRACT")
    cols = []
    for f in t.file_schema:
        if f.name in tbl.column_names:
            c = tbl[f.name]
            if not c.type.equals(f.type):
                c = c.cast(f.type)
            if not f.nullable and c.null_count:
                raise SchemaError(f"{name}.{f.name}: {c.null_count} nulls in a non-nullable column")
            cols.append(c)
        elif f.nullable:
            cols.append(pa.nulls(len(tbl), f.type))
        else:
            raise SchemaError(f"{name}: required column {f.name!r} missing")
    return pa.Table.from_arrays(cols, schema=t.file_schema)


class Writer:
    """Stream batches into one table (or one partition of it); commit() swaps it in atomically."""

    def __init__(self, name: str, partition: str | None = None, origin: str = "", rows_per_file: int = 5_000_000):
        t = CONTRACT[name]
        if bool(t.partition) != (partition is not None):
            raise SchemaError(f"{name}: " + (f"needs a {t.partition} partition" if t.partition else "is not partitioned"))
        self.name, self.partition, self.origin, self.rows_per_file = name, partition, origin, rows_per_file
        self.final = table_dir(name) / (f"{t.partition}={partition}" if partition is not None else "")
        self.tmp = root() / "tmp" / f"{name}-{partition}-{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True)
        self.rows, self._w, self._file_rows, self._n = 0, None, 0, 0

    def write(self, data) -> None:
        tbl = conform(self.name, data, self.partition)
        if not len(tbl):
            return
        if self._w is None or self._file_rows >= self.rows_per_file:
            self._close_file()
            self._w = pq.ParquetWriter(self.tmp / f"part-{self._n:05d}.parquet", CONTRACT[self.name].file_schema,
                                       compression="zstd")
            self._n += 1
        self._w.write_table(tbl, row_group_size=256_000)
        self.rows += len(tbl)
        self._file_rows += len(tbl)

    def _close_file(self):
        if self._w is not None:
            self._w.close()
            self._w, self._file_rows = None, 0

    def commit(self) -> int:
        self._close_file()
        self.final.parent.mkdir(parents=True, exist_ok=True)
        old = self.final.with_name(self.final.name + f".old-{uuid.uuid4().hex[:8]}") if self.final.exists() else None
        if old:
            self.final.rename(old)
        self.tmp.rename(self.final)
        if old:
            shutil.rmtree(old, ignore_errors=True)
        with contextlib.suppress(OSError):  # another writer may be using tmp/ right now
            (root() / "tmp").rmdir()
        with _locked():
            self._record()
        return self.rows

    def _record(self) -> None:
        m = manifest()
        entry = m["tables"].setdefault(self.name, {})
        info = {"rows": self.rows, "bytes": sum(p.stat().st_size for p in self.final.glob("*.parquet")),
                "written_at": _now(), "origin": self.origin}
        if self.partition is None:
            entry.update(info)
        else:
            entry.setdefault("partitions", {})[self.partition] = info
            entry["rows"] = sum(p["rows"] for p in entry["partitions"].values())
            entry["bytes"] = sum(p["bytes"] for p in entry["partitions"].values())
            m["sources"][self.partition] = SOURCES.get(self.partition, self.origin)
        _save_manifest(m)

    def abort(self) -> None:
        self._close_file()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.commit()
        else:
            self.abort()


def write(name: str, data, partition: str | None = None, origin: str = "") -> int:
    """Replace a whole table (or one partition) with `data`."""
    with Writer(name, partition, origin) as w:
        w.write(data)
    return w.rows


def drop(name: str, partition: str | None = None) -> None:
    t = CONTRACT[name]
    d = table_dir(name) / (f"{t.partition}={partition}" if partition is not None else "")
    shutil.rmtree(d, ignore_errors=True)
    with _locked():
        _drop_record(name, partition)


def _drop_record(name: str, partition: str | None) -> None:
    m = manifest()
    if partition is None:
        m["tables"].pop(name, None)
    elif name in m["tables"]:
        parts = m["tables"][name].get("partitions", {})
        parts.pop(partition, None)
        m["tables"][name]["rows"] = sum(p["rows"] for p in parts.values())
        m["tables"][name]["bytes"] = sum(p["bytes"] for p in parts.values())
    _save_manifest(m)


def add_license(name: str, text: str) -> None:
    (root() / "LICENSES").mkdir(parents=True, exist_ok=True)
    (root() / "LICENSES" / f"{name}.txt").write_text(text)


# ------------------------------------------------------------------ validation / info / packing
def check(verbose: bool = True) -> list[str]:
    """Validate files against the contract, manifest counts, primary keys and references."""
    problems, warn = [], []
    m = open_db()
    for name, t in CONTRACT.items():
        if not has(name):
            (problems if t.required else warn).append(f"{name}: missing")
            continue
        n = 0
        for f in table_dir(name).rglob("*.parquet"):
            fs = pq.read_schema(f)
            for fld in fs:
                if fld.name not in t.file_schema.names:
                    problems.append(f"{name}: {f.name} has unknown column {fld.name}")
                elif not fld.type.equals(t.file_schema.field(fld.name).type):
                    problems.append(f"{name}: {f.name} column {fld.name} is {fld.type}, "
                                    f"contract says {t.file_schema.field(fld.name).type}")
            for fld in t.file_schema:
                if fld.name not in fs.names and not fld.nullable:
                    problems.append(f"{name}: {f.name} lacks required column {fld.name}")
            n += pq.read_metadata(f).num_rows
        listed = m["tables"].get(name, {}).get("rows")
        if listed is not None and listed != n:
            problems.append(f"{name}: manifest says {listed:,} rows, files hold {n:,}")
        if t.pk and n > PK_CHECK_MAX_ROWS:
            warn.append(f"{name}: primary-key check skipped ({n:,} rows > {PK_CHECK_MAX_ROWS:,})")
        elif t.pk:
            keys = read_arrow(name, columns=t.pk)
            dup = len(keys) - len(keys.group_by(t.pk).aggregate([]))
            if dup:
                problems.append(f"{name}: {dup:,} duplicate primary keys {t.pk}")
        if verbose:
            print(f"[musicdb] ok? {name:20s} {n:>12,} rows")
    if has("lastfm_tracks") and has("similars"):
        n = len(read_arrow("lastfm_tracks", ["tid"]))
        tids = read_arrow("lastfm_tracks", ["tid"])["tid"]
        if pc.min(tids).as_py() != 0 or pc.max(tids).as_py() != n - 1:
            problems.append("lastfm_tracks: tid must be dense 0..n-1")
        s = read_arrow("similars", ["tid", "sim_tid"])
        for c in ("tid", "sim_tid"):
            if pc.max(s[c]).as_py() >= n or pc.min(s[c]).as_py() < 0:
                problems.append(f"similars.{c}: references outside lastfm_tracks")
    for p in problems + warn:
        print(("[musicdb] PROBLEM " if p in problems else "[musicdb] warning ") + p)
    if verbose:
        print(f"[musicdb] {'OK' if not problems else f'{len(problems)} problem(s)'} - schema v{m['schema_version']}, "
              f"{size_mb():.0f} MB at {root()}")
    return problems


def size_mb() -> float:
    return sum(p.stat().st_size for p in root().rglob("*") if p.is_file()) / 2**20


def info() -> None:
    m = open_db()
    print(f"musicdb schema v{m['schema_version']} at {root()}  ({size_mb():.0f} MB, updated {m.get('updated_at')})")
    for name, t in CONTRACT.items():
        e = m["tables"].get(name)
        if not e:
            print(f"  {name:20s} {'-':>12s}            {t.doc}")
            continue
        print(f"  {name:20s} {e['rows']:>12,} {e['bytes'] / 2**20:7.0f} MB  {t.doc}")
        for p, pe in sorted(e.get("partitions", {}).items()):
            print(f"    {t.partition}={p:24s} {pe['rows']:>12,} {pe['bytes'] / 2**20:7.0f} MB")


def pack(out: Path | None = None) -> Path:
    """One file to copy to another machine (parquet is already compressed, so plain tar)."""
    check(verbose=False)
    out = Path(out or root().parent / f"musicdb-v{SCHEMA_VERSION}.tar")
    with tarfile.open(out, "w") as tar:
        tar.add(root(), arcname="musicdb", filter=lambda ti: None if "/tmp" in ti.name else ti)
    print(f"[musicdb] packed {out} ({out.stat().st_size / 2**20:.0f} MB)")
    return out


def unpack(file: Path) -> None:
    file = Path(file)
    with tarfile.open(file) as tar:
        names = tar.getnames()
        if not names or any(not (n == "musicdb" or n.startswith("musicdb/")) for n in names):
            raise SystemExit(f"{file} is not a musicdb archive")
        if root().exists():
            raise SystemExit(f"{root()} already exists; move it away first")
        tar.extractall(root().parent, filter="data")
    if root().name != "musicdb":
        (root().parent / "musicdb").rename(root())
    check()

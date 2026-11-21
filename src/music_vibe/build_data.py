"""Offline step (Spark): catalogue + tag weights.

Inputs : data/lb/*.parquet (ListenBrainz sample) + Last.fm cache (optional)
Outputs: artifacts/tracks.parquet      one row per song (duplicates merged), popularity
         artifacts/tags.parquet        vibe vocabulary with document frequency / idf
         artifacts/track_tags.parquet  top-k weighted tags per track

Tag weight = sum over sources of source_weight * (count / max count of that track+source),
then multiplied by idf so ubiquitous tags ("rock", "pop") don't dominate.
"""

from __future__ import annotations

import re
import shutil
import time

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from .config import (ARTIFACTS, CATALOG_SIZE, DATA, LB, MIN_TAG_TRACKS, SOURCE_WEIGHTS,
                     TOP_TAGS_PER_TRACK, is_vibe_tag, normalize_tag)
from .lastfm import iter_cache
from .spark_env import get_spark

# versions that make poor recommendations: demos, outtakes, skits, live/instrumental takes
JUNK_TITLE = (r"(?i)\b(demo|outtake|interview|commentary|karaoke|rehearsal|skit|intro|outro|interlude"
              r"|reprise|acapella|a cappella)\b|[\(\[-].*\b(live|instrumental|remix|mix|version|take)\b")

# song-identity normalization, shared by Python (Last.fm side) and Spark (catalogue side)
_NORM_STEPS = [
    (r"\s*[\(\[][^\)\]]*[\)\]]", ""),                  # (Remastered 2011), [Live]
    (r"\s+-\s+(remaster|live|mono|stereo|single|radio|edit|version).*$", ""),
    (r"\s+(feat\.?|ft\.?|featuring|with)\s.*$", ""),  # feat. X
    (r"[^\w\s]", ""),
    (r"\s+", " "),
]


def norm_py(s: str) -> str:
    s = (s or "").lower()
    for pat, rep in _NORM_STEPS:
        s = re.sub(pat, rep, s)
    return s.strip()


def norm_col(c):
    c = F.lower(c)
    for pat, rep in _NORM_STEPS:
        c = F.regexp_replace(c, "(?U)" + pat, rep)
    return F.trim(c)


def export_lastfm() -> pd.DataFrame:
    """Flatten cached Last.fm responses into (recording_mbid | key, tag, count)."""
    rows = []
    for key, data in iter_cache("toptags|"):
        mbid = key.split("|", 1)[1]
        for t in (data.get("toptags") or {}).get("tag", []) or []:
            if int(t.get("count", 0)) > 0:
                rows.append((mbid, None, t["name"], float(t["count"])))
    for key, data in iter_cache("seed|"):
        _, tag, page, per = key.split("|")
        for i, tr in enumerate((data.get("tracks") or {}).get("track", []) or []):
            rank = (int(page) - 1) * int(per) + i
            score = max(100.0 * (1 - rank / 600), 20.0)  # top of the tag chart counts most
            rows.append((None, norm_py(tr["artist"]["name"]) + "|" + norm_py(tr["name"]), tag, score))
    df = pd.DataFrame(rows, columns=["recording_mbid", "song_key", "tag", "count"]).astype(
        {"recording_mbid": "string", "song_key": "string", "tag": "string", "count": "float64"})
    print(f"[data] Last.fm rows: {len(df):,} ({df.recording_mbid.notna().sum():,} track-tags, "
          f"{df.song_key.notna().sum():,} seed-chart entries)")
    return df


def run(catalog_size: int = CATALOG_SIZE, top_tags: int = TOP_TAGS_PER_TRACK,
        min_tag_tracks: int = MIN_TAG_TRACKS, driver_memory: str = "3g") -> None:
    t0 = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lfm = export_lastfm()
    lfm_path = DATA / "lastfm_tags.parquet"
    lfm.to_parquet(lfm_path, index=False)

    spark = get_spark("music-vibe-build-data", driver_memory=driver_memory)
    try:
        rec = spark.read.parquet(str(LB / "recordings.parquet")).where(
            F.col("title").isNotNull() & ~F.col("title").rlike(JUNK_TITLE))
        pop = (spark.read.parquet(str(LB / "popularity.parquet")).groupBy("recording_mbid")
               .agg(F.sum("total_listen_count").alias("listens"), F.sum("total_user_count").alias("users")))
        length = spark.read.parquet(str(LB / "length.parquet"))

        # ---- catalogue: one row per song, most popular recording wins ----
        cat = (rec.join(pop, "recording_mbid").join(length, "recording_mbid", "left")
               .withColumn("song_key", F.concat_ws("|", norm_col(F.col("artist")), norm_col(F.col("title")))))
        w_song = Window.partitionBy("song_key").orderBy(F.desc("users"), F.desc("listens"))
        song_totals = cat.groupBy("song_key").agg(F.sum("users").alias("song_users"),
                                                  F.sum("listens").alias("song_listens"),
                                                  F.collect_set("recording_mbid").alias("all_mbids"))
        cat = (cat.withColumn("r", F.row_number().over(w_song)).where("r = 1").drop("r", "users", "listens")
               .join(song_totals, "song_key")
               .orderBy(F.desc("song_users")).limit(catalog_size).cache())
        n_cat = cat.count()

        # every recording mbid of a kept song -> its canonical track
        alias = cat.select(F.col("recording_mbid").alias("track_mbid"), F.explode("all_mbids").alias("recording_mbid"))

        # ---- tags from MusicBrainz (via any alias recording) and Last.fm ----
        mb = (spark.read.parquet(str(LB / "mb_tags.parquet")).join(F.broadcast(alias), "recording_mbid")
              .select("track_mbid", "tag", F.col("count").cast("double").alias("count"), "source"))
        lf = spark.read.parquet(str(lfm_path))
        lf_direct = lf.where(F.col("recording_mbid").isNotNull()).join(F.broadcast(alias), "recording_mbid") \
            .select("track_mbid", "tag", "count", F.lit("lastfm").alias("source"))
        lf_seed = lf.where(F.col("song_key").isNotNull()).join(
            F.broadcast(cat.select(F.col("recording_mbid").alias("track_mbid"), "song_key")), "song_key") \
            .select("track_mbid", "tag", "count", F.lit("lastfm").alias("source"))
        tags = mb.unionByName(lf_direct).unionByName(lf_seed).where(F.col("count") > 0)

        # normalize tag strings in Python (small distinct set), keep vibe tags only
        distinct = [r.tag for r in tags.select("tag").distinct().collect()]
        mapping = pd.DataFrame([(t, normalize_tag(t)) for t in distinct if t and is_vibe_tag(t)],
                               columns=["tag", "ntag"])
        tags = tags.join(F.broadcast(spark.createDataFrame(mapping)), "tag") \
            .groupBy("track_mbid", "ntag", "source").agg(F.max("count").alias("count"))

        # per track+source scaling, then weighted sum over sources
        w_src = Window.partitionBy("track_mbid", "source")
        sw = F.create_map(*[x for k, v in SOURCE_WEIGHTS.items() for x in (F.lit(k), F.lit(v))])
        scored = (tags.withColumn("rel", F.col("count") / F.max("count").over(w_src))
                  .withColumn("w", F.col("rel") * sw[F.col("source")])
                  .groupBy("track_mbid", "ntag").agg(F.sum("w").alias("score"),
                                                     F.collect_set("source").alias("sources")))

        df = scored.groupBy("ntag").agg(F.countDistinct("track_mbid").alias("df")) \
            .where(F.col("df") >= min_tag_tracks) \
            .withColumn("idf", F.log(F.lit(float(n_cat)) / F.col("df")))
        weighted = scored.join(F.broadcast(df), "ntag").withColumn("weight", F.col("score") * F.col("idf"))
        w_rank = Window.partitionBy("track_mbid").orderBy(F.desc("weight"))
        top = weighted.withColumn("rank", F.row_number().over(w_rank)).where(F.col("rank") <= top_tags)

        track_tags = top.select(F.col("track_mbid").alias("recording_mbid"), F.col("ntag").alias("tag"),
                                F.col("weight").cast("float"), F.col("rank").cast("short"),
                                F.array_contains("sources", "lastfm").alias("from_lastfm")).toPandas()
        tag_pd = df.withColumnRenamed("ntag", "tag").toPandas().sort_values("df", ascending=False)
        tracks = cat.select("recording_mbid", "title", "artist", "artist_mbids", "release", "year",
                            "release_group_mbid", "caa_id", "caa_release_mbid", "length",
                            F.col("song_users").alias("users"), F.col("song_listens").alias("listens"),
                            F.size("all_mbids").alias("n_recordings")).toPandas()
        aliases = alias.toPandas()
    finally:
        spark.stop()
        shutil.rmtree(ARTIFACTS / "spark-tmp", ignore_errors=True)

    tracks = tracks[tracks.recording_mbid.isin(set(track_tags.recording_mbid))].reset_index(drop=True)
    tracks.to_parquet(ARTIFACTS / "tracks.parquet", index=False)
    aliases[aliases.track_mbid.isin(set(tracks.recording_mbid))].to_parquet(ARTIFACTS / "aliases.parquet", index=False)
    tag_pd.to_parquet(ARTIFACTS / "tags.parquet", index=False)
    track_tags.to_parquet(ARTIFACTS / "track_tags.parquet", index=False)
    lf_share = track_tags.groupby("recording_mbid").from_lastfm.any().mean()
    print(f"[data] {len(tracks):,} tracks (of {n_cat:,} songs) | {len(tag_pd):,} tags | "
          f"{len(track_tags):,} track-tags | {lf_share:.0%} tracks with Last.fm tags | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()

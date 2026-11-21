"""Offline step (Spark): catalogue + tag weights. Reads only musicdb (+ the kNN tag artifact).

Inputs : musicdb tables recordings, recording_lengths, popularity, tags (all sources), lastfm_tracks
         artifacts/knn_tags.parquet (kNN-predicted tags, optional - `make knn`)
         artifacts/audio_tags.parquet (tags predicted from AcousticBrainz audio, optional - `make audio`)
Outputs: artifacts/tracks.parquet        one row per song (duplicates merged), popularity
         artifacts/tags.parquet          vibe vocabulary with document frequency / idf
         artifacts/track_tags.parquet    top-k weighted tags per track
         artifacts/aliases.parquet       every recording MBID of a kept song -> its canonical track
         artifacts/lastfm_match.parquet  catalogue track -> Last.fm dataset track + split (for evaluation)

Last.fm dataset TEST-split tags never enter the index; test tracks only get kNN-predicted tags.

Tag weight = sum over sources of source_weight * (count / max count of that track+source),
then multiplied by idf so ubiquitous tags ("rock", "pop") don't dominate. kNN-predicted tags keep
their absolute 0-1 score instead (it already includes the prediction's confidence); same for audio tags.
"""

from __future__ import annotations

import shutil
import time

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from . import musicdb as db
from .audio_tags import AUDIO_TAGS
from .config import (API_SOURCES, ARTIFACTS, CATALOG_SIZE, LASTFM_SOURCES, MIN_TAG_TRACKS, PREDICTED_SOURCES,
                     SOURCE_WEIGHTS, TOP_TAGS_PER_TRACK, is_vibe_tag, normalize_tag)
from .knn_tags import KNN_TAGS
from .spark_env import get_spark

# versions that make poor recommendations: demos, outtakes, skits, live/instrumental takes
JUNK_TITLE = (r"(?i)\b(demo|outtake|interview|commentary|karaoke|rehearsal|skit|intro|outro|interlude"
              r"|reprise|acapella|a cappella)\b|[\(\[-].*\b(live|instrumental|remix|mix|version|take)\b")
RECORDING_SOURCES = ("mb_recording", "mb_release_group", "mb_artist", "lastfm_api")  # item_id = recording MBID


def run(catalog_size: int = CATALOG_SIZE, top_tags: int = TOP_TAGS_PER_TRACK,
        min_tag_tracks: int = MIN_TAG_TRACKS, driver_memory: str = "3g") -> None:
    t0 = time.time()
    db.open_db()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    spark = get_spark("moodplay-build-data", driver_memory=driver_memory)

    def table(name):
        return spark.read.option("mergeSchema", "true").parquet(db.path(name))

    try:
        rec = table("recordings").select("recording_mbid", "title", "artist", "artist_mbids", "release", "year",
                                         "release_group_mbid", "caa_id", "caa_release_mbid", "song_key") \
            .where(F.col("title").isNotNull() & F.col("song_key").isNotNull() & ~F.col("title").rlike(JUNK_TITLE))
        pop = (table("popularity").where((F.col("source") == "listenbrainz") & (F.col("entity_type") == "recording"))
               .groupBy(F.col("entity_id").alias("recording_mbid"))
               .agg(F.sum("listens").alias("listens"), F.sum("users").alias("users")))
        length = table("recording_lengths").select("recording_mbid", "length")

        # ---- catalogue: one row per song, most popular recording wins ----
        cat = rec.join(pop, "recording_mbid").join(length, "recording_mbid", "left")
        w_song = Window.partitionBy("song_key").orderBy(F.desc("users"), F.desc("listens"))
        song_totals = cat.groupBy("song_key").agg(F.sum("users").alias("song_users"),
                                                  F.sum("listens").alias("song_listens"),
                                                  F.collect_set("recording_mbid").alias("all_mbids"))
        cat = (cat.withColumn("r", F.row_number().over(w_song)).where("r = 1").drop("r", "users", "listens")
               .join(song_totals, "song_key")
               .orderBy(F.desc("song_users")).limit(catalog_size).cache())
        n_cat = cat.count()
        songs = F.broadcast(cat.select(F.col("recording_mbid").alias("track_mbid"), "song_key"))

        # every recording mbid of a kept song -> its canonical track
        alias = cat.select(F.col("recording_mbid").alias("track_mbid"), F.explode("all_mbids").alias("recording_mbid"))

        # ---- tags: by recording MBID (MusicBrainz, Last.fm API), by song (seed charts) ----
        all_tags = table("tags").select("item_id", "tag", F.col("count").cast("double").alias("count"), "source")
        by_rec = all_tags.where(F.col("source").isin(*RECORDING_SOURCES)) \
            .join(F.broadcast(alias.withColumnRenamed("recording_mbid", "item_id")), "item_id")
        by_song = all_tags.where(F.col("source") == "lastfm_seed") \
            .join(songs.withColumnRenamed("song_key", "item_id"), "item_id")
        tags = by_rec.select("track_mbid", "tag", "count", "source") \
            .unionByName(by_song.select("track_mbid", "tag", "count", "source"))

        # ---- Last.fm dataset: train-split tags + kNN-predicted tags (test tracks, untagged tracks) ----
        match = None
        if db.has("lastfm_tracks"):
            lf_tracks = table("lastfm_tracks").select("tid", "track_id", "split", "song_key")
            match = lf_tracks.join(songs, "song_key").select("track_mbid", "tid", "track_id", "split").cache()
            train_ids = match.where(F.col("split") == "train").select("track_mbid", F.col("track_id").alias("item_id"))
            ds = all_tags.where(F.col("source") == "lastfm_ds").join(train_ids, "item_id")
            tags = tags.unionByName(ds.select("track_mbid", "tag", "count", "source"))
            if KNN_TAGS.exists():
                knn = spark.read.parquet(str(KNN_TAGS)).join(match.select("track_mbid", "tid"), "tid") \
                    .select("track_mbid", "tag", F.col("count").cast("double").alias("count"),
                            F.lit("lastfm_knn").alias("source"))
                tags = tags.unionByName(knn)
            else:
                print("[data] no artifacts/knn_tags.parquet - run `make knn` to add predicted tags")
        else:
            print("[data] musicdb has no Last.fm dataset tables")
        if AUDIO_TAGS.exists():
            audio = spark.read.parquet(str(AUDIO_TAGS)).withColumnRenamed("recording_mbid", "item_id") \
                .join(F.broadcast(alias.withColumnRenamed("recording_mbid", "item_id")), "item_id") \
                .select("track_mbid", "tag", F.col("count").cast("double").alias("count"), F.lit("audio").alias("source"))
            tags = tags.unionByName(audio)
        else:
            print("[data] no artifacts/audio_tags.parquet - run `make audio` to add audio-predicted tags")
        tags = tags.where(F.col("count") > 0)

        # normalize tag strings in Python (small distinct set), keep vibe tags only
        distinct = [r.tag for r in tags.select("tag").distinct().collect()]
        mapping = pd.DataFrame([(t, normalize_tag(t)) for t in distinct if t and is_vibe_tag(t)],
                               columns=["tag", "ntag"])
        tags = tags.join(F.broadcast(spark.createDataFrame(mapping)), "tag") \
            .groupBy("track_mbid", "ntag", "source").agg(F.max("count").alias("count"))

        # per track+source scaling, then weighted sum over sources
        w_src = Window.partitionBy("track_mbid", "source")
        sw = F.create_map(*[x for k, v in SOURCE_WEIGHTS.items() for x in (F.lit(k), F.lit(v))])
        rel = F.when(F.col("source").isin(*PREDICTED_SOURCES), F.col("count") / 100.0) \
            .otherwise(F.col("count") / F.max("count").over(w_src))
        scored = (tags.withColumn("rel", rel)
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
                                F.arrays_overlap("sources", F.array(*map(F.lit, LASTFM_SOURCES))).alias("from_lastfm"),
                                F.arrays_overlap("sources", F.array(*map(F.lit, API_SOURCES))).alias("from_api")).toPandas()
        api_tracks = {r.track_mbid for r in
                      tags.where(F.col("source").isin(*API_SOURCES)).select("track_mbid").distinct().collect()}
        matched = match.toPandas() if match is not None else None
        tag_pd = df.withColumnRenamed("ntag", "tag").toPandas().sort_values("df", ascending=False)
        n_audio = tags.where(F.col("source") == "audio").select("track_mbid").distinct().count()
        tracks = cat.select("recording_mbid", "title", "artist", "artist_mbids", "release", "year", "song_key",
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
    if matched is not None:
        matched = matched.rename(columns={"track_mbid": "recording_mbid"}).drop(columns="track_id")
        matched = matched[matched.recording_mbid.isin(set(tracks.recording_mbid))]
        matched["has_api"] = matched.recording_mbid.isin(api_tracks)
        matched.to_parquet(ARTIFACTS / "lastfm_match.parquet", index=False)
        splits = matched.groupby("recording_mbid").split.agg(lambda s: "train" if (s == "train").any() else "test")
        print(f"[data] Last.fm dataset matches {len(splits):,} tracks "
              f"({(splits == 'train').sum():,} train, {(splits == 'test').sum():,} test-only)")
    lf_share = track_tags.groupby("recording_mbid").from_lastfm.any().mean()
    print(f"[data] {len(tracks):,} tracks (of {n_cat:,} songs) | {len(tag_pd):,} tags | "
          f"{len(track_tags):,} track-tags | {lf_share:.0%} tracks with Last.fm tags | "
          f"{n_audio / max(len(tracks), 1):.0%} with audio tags | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()

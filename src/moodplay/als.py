"""Implicit-feedback ALS (Spark) on musicdb `feedback` (ListenBrainz loved/hated, web-app thumbs) and
musicdb `plays` (Echo Nest Taste Profile play counts, linked to the catalogue by song_key).

ListenBrainz feedback is ~98% "loved", so explicit-rating ALS has nothing to learn
from. Implicit ALS (Hu, Koren & Volinsky 2008) models preference p in {0,1} with
confidence c = 1 + alpha*r: a thumbs-up is p=1 (r=1), a thumbs-down is a confident p=0, a play
count is p=1 with r=log(1+plays), and everything unseen is a weak p=0.

Evaluation: hold out 10% of the likes of ListenBrainz-sample users with >= 5 likes (the same population
whatever extra sources are used, so numbers stay comparable), rank all tracks for that user, and report
hit-rate@50 against a most-liked-tracks baseline.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

from . import musicdb as db
from .config import ARTIFACTS

ALS_FILE = ARTIFACTS / "als_items.npy"
ALS_META = ARTIFACTS / "als_meta.json"


def _interactions() -> pd.DataFrame:
    """All feedback sources; user ids are only unique within a source, so the user key is (source, user_id)."""
    db.open_db()
    aliases = pd.read_parquet(ARTIFACTS / "aliases.parquet")
    fb = db.read("feedback", ["user_id", "recording_mbid", "value", "source"])
    fb = fb.merge(aliases, on="recording_mbid")  # any recording of a song -> its catalogue track
    fb["user_id"] = fb.source.astype(str) + ":" + fb.user_id.astype(str)
    fb["value"] = fb.value.astype("int8")
    fb["value"] = fb.value.astype("float32")
    return fb.groupby(["user_id", "track_mbid"], as_index=False).agg(value=("value", "last"), source=("source", "first"))


def _plays() -> pd.DataFrame:
    """Taste Profile play counts on catalogue songs (MSD song -> song_key -> catalogue track)."""
    if not db.has("plays", "msd_taste") or not db.has("msd_tracks"):
        return pd.DataFrame(columns=["user_id", "track_mbid", "value", "source"])
    tracks = pd.read_parquet(ARTIFACTS / "tracks.parquet", columns=["recording_mbid", "song_key"])
    msd = db.read("msd_tracks", ["song_id", "song_key", "mismatch"])
    msd = msd[~msd.mismatch].merge(tracks, on="song_key")[["song_id", "recording_mbid"]].drop_duplicates("song_id")
    pl = db.read("plays", ["user_id", "item_id", "plays"], sources=["msd_taste"],
                 filter=pads.field("item_id").isin(pa.array(msd.song_id.to_numpy(), pa.string())))
    pl = pl.merge(msd, left_on="item_id", right_on="song_id")
    pl = pl.groupby(["user_id", "recording_mbid"], as_index=False).plays.sum()
    return pd.DataFrame({"user_id": "msd_taste:" + pl.user_id.astype(str), "track_mbid": pl.recording_mbid,
                         "value": np.log1p(pl.plays.to_numpy()).astype("float32"), "source": "msd_taste"})


def run(rank: int = 48, max_iter: int = 15, reg: float = 0.1, alpha: float = 20.0, holdout: float = 0.1,
        seed: int = 42, driver_memory: str = "3g", plays: bool = False, min_user_items: int = 2) -> dict:
    from pyspark.ml.recommendation import ALS

    from .spark_env import get_spark

    t0 = time.time()
    idx = np.load(ARTIFACTS / "index.npz", allow_pickle=False)
    track_mbids = idx["track_mbids"]
    item_of = {m: i for i, m in enumerate(track_mbids)}

    df = _interactions()
    if plays:
        p = _plays()
        p = p[p.track_mbid.isin(item_of)]
        n_items = p.groupby("user_id").size()  # one-song users add nothing to the factors
        p = p[p.user_id.isin(n_items[n_items >= min_user_items].index)]
        df = pd.concat([df, p], ignore_index=True)
    df = df[df.track_mbid.isin(item_of)].reset_index(drop=True)
    df["item"] = df.track_mbid.map(item_of).astype("int32")
    users = {u: i for i, u in enumerate(df.user_id.unique())}
    df["user"] = df.user_id.map(users).astype("int32")
    n_app = int((df.source == "app").sum())
    n_plays = int((df.source == "msd_taste").sum())

    # hold out some likes of users with >= 5 likes
    rng = np.random.default_rng(seed)
    likes = df[(df.value > 0) & (df.source == "listenbrainz")]
    counts = likes.groupby("user").size()
    eligible = likes[likes.user.isin(counts[counts >= 5].index)]
    test = eligible.sample(frac=holdout, random_state=seed) if holdout > 0 else eligible.iloc[:0]
    train = df.drop(test.index)

    spark = get_spark("moodplay-als", driver_memory=driver_memory)
    try:
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        sdf = spark.createDataFrame(train[["user", "item"]].assign(value=train.value.astype("float32")))
        model = ALS(rank=rank, maxIter=max_iter, regParam=reg, alpha=alpha, implicitPrefs=True,
                    userCol="user", itemCol="item", ratingCol="value", nonnegative=False,
                    coldStartStrategy="drop", seed=seed, checkpointInterval=-1).fit(sdf)
        items = model.itemFactors.toPandas()
        ufac = model.userFactors.toPandas()
    finally:
        spark.stop()

    Y = np.zeros((len(track_mbids), rank), dtype=np.float32)
    Y[items.id.to_numpy()] = np.stack(items.features.to_numpy()).astype(np.float32)
    np.save(ALS_FILE, Y)

    # ---- evaluation: hit-rate@50 on held-out likes ----
    hr, hr_pop = None, None
    if len(test):
        U = np.zeros((len(users), rank), dtype=np.float32)
        U[ufac.id.to_numpy()] = np.stack(ufac.features.to_numpy())
        lb_likes = train[(train.value > 0) & (train.source == "listenbrainz")]  # same baseline with or without plays
        pop = np.bincount(lb_likes.item, minlength=len(track_mbids)).astype(np.float32)
        seen = train[train.user.isin(test.user.unique())].groupby("user").item.apply(np.array).to_dict()
        hits = hits_pop = 0
        for u, grp in test.groupby("user"):
            s = Y @ U[u]
            s[seen.get(u, [])] = -np.inf
            p = pop.copy()
            p[seen.get(u, [])] = -np.inf
            top, top_p = set(np.argpartition(-s, 50)[:50]), set(np.argpartition(-p, 50)[:50])
            hits += sum(i in top for i in grp.item)
            hits_pop += sum(i in top_p for i in grp.item)
        hr, hr_pop = hits / len(test), hits_pop / len(test)

    meta = {"rank": rank, "reg": reg, "alpha": alpha, "max_iter": max_iter, "implicit": True,
            "interactions": int(len(train)), "users": len(users), "items_with_factors": int(Y.any(axis=1).sum()),
            "app_feedback_rows": n_app, "taste_profile_rows": n_plays, "test_likes": int(len(test)),
            "hit_rate_at_50": None if hr is None else round(hr, 4),
            "hit_rate_at_50_popularity": None if hr_pop is None else round(hr_pop, 4),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0)}
    ALS_META.write_text(json.dumps(meta, indent=2))
    print("[als]", json.dumps(meta))
    return meta


def fold_in(Y: np.ndarray, YtY: np.ndarray, rows: list[int], prefs: list[int], reg: float, alpha: float):
    """Implicit-ALS user solve: (YtY + Y_S^T (C-I) Y_S + reg I) u = Y_S^T C p."""
    Ys = Y[rows]
    c = 1.0 + alpha
    A = YtY + (c - 1.0) * Ys.T @ Ys + reg * np.eye(Y.shape[1], dtype=np.float32)
    p = np.array([1.0 if v > 0 else 0.0 for v in prefs], dtype=np.float32)
    b = Ys.T @ (c * p)
    return np.linalg.solve(A, b).astype(np.float32)

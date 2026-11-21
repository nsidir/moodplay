"""Implicit-feedback ALS (Spark) on ListenBrainz loved/hated tracks + web-app thumbs.

ListenBrainz feedback is ~98% "loved", so explicit-rating ALS has nothing to learn
from. Implicit ALS (Hu, Koren & Volinsky 2008) models preference p in {0,1} with
confidence c = 1 + alpha: a thumbs-up is p=1, a thumbs-down is a confident p=0, and
everything unseen is a weak p=0.

Evaluation: hold out 10% of each eligible user's likes, rank all tracks for that
user, and report hit-rate@50 against a most-liked-tracks baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS, LB

APP_USER_OFFSET = 50_000_000
ALS_FILE = ARTIFACTS / "als_items.npy"
ALS_META = ARTIFACTS / "als_meta.json"


def _interactions(app_feedback: Path | None) -> pd.DataFrame:
    aliases = pd.read_parquet(ARTIFACTS / "aliases.parquet")
    lb = pd.read_parquet(LB / "feedback.parquet").rename(columns={"feedback": "value"})
    lb = lb.merge(aliases, on="recording_mbid")[["user_id", "track_mbid", "value"]]
    parts = [lb]
    if app_feedback and Path(app_feedback).exists():
        app = pd.read_parquet(app_feedback)
        if len(app):
            app = app.rename(columns={"recording_mbid": "track_mbid"})
            app["user_id"] = app.user_id + APP_USER_OFFSET
            parts.append(app[["user_id", "track_mbid", "value"]])
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["user_id", "track_mbid"], as_index=False).value.last()


def run(app_feedback: Path | None = None, rank: int = 48, max_iter: int = 15, reg: float = 0.1,
        alpha: float = 20.0, holdout: float = 0.1, seed: int = 42, driver_memory: str = "3g") -> dict:
    from pyspark.ml.recommendation import ALS

    from .spark_env import get_spark

    t0 = time.time()
    idx = np.load(ARTIFACTS / "index.npz", allow_pickle=False)
    track_mbids = idx["track_mbids"]
    item_of = {m: i for i, m in enumerate(track_mbids)}

    df = _interactions(app_feedback)
    df = df[df.track_mbid.isin(item_of)]
    df["item"] = df.track_mbid.map(item_of).astype("int32")
    users = {u: i for i, u in enumerate(df.user_id.unique())}
    df["user"] = df.user_id.map(users).astype("int32")
    n_app = int((df.user_id >= APP_USER_OFFSET).sum())

    # hold out some likes of users with >= 5 likes
    rng = np.random.default_rng(seed)
    likes = df[df.value > 0]
    counts = likes.groupby("user").size()
    eligible = likes[likes.user.isin(counts[counts >= 5].index)]
    test = eligible.sample(frac=holdout, random_state=seed) if holdout > 0 else eligible.iloc[:0]
    train = df.drop(test.index)

    spark = get_spark("music-vibe-als", driver_memory=driver_memory)
    try:
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
        pop = np.bincount(train[train.value > 0].item, minlength=len(track_mbids)).astype(np.float32)
        seen = train.groupby("user").item.apply(np.array).to_dict()
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
            "app_feedback_rows": n_app, "test_likes": int(len(test)),
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

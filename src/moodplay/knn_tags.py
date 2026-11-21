"""kNN tag model trained on the Last.fm dataset TRAIN split, evaluated on the TEST split.

Inputs (musicdb): lastfm_tracks (official split), tags/source=lastfm_ds, similars.

train     tags(t) = sum_n s(t,n) * tags_train(n) / (sum_n s(t,n) + KNN_SHRINK)
          over t's SIMILARS_PER_TRACK strongest similar tracks n that have train tags. Applied to every
          track without train tags (all test tracks + untagged train tracks) -> artifacts/knn_tags.parquet.
          Similarity comes from listening data, so it is a feature, not a label: no test tags are used.
evaluate  compare predictions on TEST tracks with their held-out Last.fm tags: precision/recall@k vs a
          most-frequent-tags baseline, plus ROC-AUC per mood tag -> artifacts/eval_tags.json

build_data joins train tags (source lastfm_ds) and predictions (source lastfm_knn) into the catalogue;
test tags never enter the index.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import scipy.sparse as sp
from scipy.stats import rankdata

from . import musicdb as db
from .config import ARTIFACTS, KNN_TOP_TAGS, MIN_TAG_TRACKS, SIMILARS_PER_TRACK, is_vibe_tag, normalize_tag

KNN_TAGS = ARTIFACTS / "knn_tags.parquet"
KNN_VOCAB = ARTIFACTS / "knn_vocab.parquet"
KNN_SHRINK = 0.5      # pseudo-similarity mass: tracks with few/weak tagged neighbours get lower confidence
TRUTH_MIN_COUNT = 10  # a held-out tag counts as ground truth at >= this Last.fm count (0-100)
MOOD_TAGS = ["chill", "mellow", "relaxing", "happy", "sad", "melancholic", "energetic", "upbeat", "romantic",
             "dark", "atmospheric", "dreamy", "aggressive", "party", "dance", "sexy", "calm", "epic",
             "psychedelic", "nostalgic", "summer", "driving", "sleep", "funky", "groovy", "acoustic"]


# ---------------------------------------------------------------- shared loading
def normalize_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Raw source tags -> normalized vibe tags (max count per item+tag)."""
    norm = {t: normalize_tag(t) for t in tags.tag.unique() if is_vibe_tag(t)}
    tags = tags[tags.tag.isin(norm.keys())]
    return tags.assign(tag=tags.tag.map(norm)).groupby(["tid", "tag"], as_index=False)["count"].max()


def _load():
    db.open_db()
    tracks = db.read("lastfm_tracks", ["tid", "track_id", "split"]).sort_values("tid").reset_index(drop=True)
    tags = db.read("tags", ["item_id", "tag", "count"], sources=["lastfm_ds"])
    tags["tid"] = pd.Index(tracks.track_id).get_indexer(tags.pop("item_id"))
    tags = tags[(tags.tid >= 0) & (tags["count"] > 0)]
    tags = normalize_tags(tags)
    test = tracks.tid[tracks.split == "test"].to_numpy()
    is_test = np.zeros(len(tracks), dtype=bool)
    is_test[test] = True
    return tracks, tags, is_test


def _tag_matrix(tags: pd.DataFrame, n: int, vocab: pd.Index) -> sp.csr_matrix:
    t = tags[tags.tag.isin(vocab)]
    return sp.csr_matrix(((t["count"].to_numpy(np.float32) / 100.0),
                          (t.tid.to_numpy(), vocab.get_indexer(t.tag))), shape=(n, len(vocab)))


# ---------------------------------------------------------------- 2. train (kNN tag propagation)
def train(min_tag_tracks: int = MIN_TAG_TRACKS, top_tags: int = KNN_TOP_TAGS, chunk: int = 50_000) -> None:
    t0 = time.time()
    tracks, tags, is_test = _load()
    n = len(tracks)
    tr_tags = tags[~is_test[tags.tid.to_numpy()]]
    df = tr_tags.groupby("tag", observed=True).tid.nunique()
    vocab = pd.Index(df[df >= min_tag_tracks].index.astype(str))
    T = _tag_matrix(tr_tags, n, vocab)                       # train split only
    has_tags = np.diff(T.indptr) > 0

    sim = db.read("similars", ["tid", "sim_tid", "score"], filter=pads.field("rank") < SIMILARS_PER_TRACK)
    sim = sim[has_tags[sim.sim_tid.to_numpy()]]               # neighbours must carry train tags
    S = sp.csr_matrix((sim.score.to_numpy(np.float32), (sim.tid.to_numpy(), sim.sim_tid.to_numpy())), shape=(n, n))
    del sim
    support = np.asarray(S.sum(axis=1)).ravel()
    targets = np.flatnonzero(~has_tags & (support > 0))      # test tracks + untagged train tracks

    parts = []
    for lo in range(0, len(targets), chunk):
        rows = targets[lo:lo + chunk]
        P = (sp.diags(1.0 / (support[rows] + KNN_SHRINK)) @ (S[rows] @ T)).tocoo()
        p = pd.DataFrame({"tid": rows[P.row].astype("int32"), "tag_i": P.col.astype("int32"),
                          "score": P.data.astype("float32")})
        p = p.sort_values(["tid", "score"], ascending=[True, False]).groupby("tid").head(top_tags)
        parts.append(p)
    pred = pd.concat(parts, ignore_index=True)
    pred["tag"] = vocab[pred.tag_i.to_numpy()]
    pred = pred.drop(columns="tag_i")
    # predictions are on the 0-1 tag scale already shrunk by confidence; keep count-like 0-100 for build_data
    pred["count"] = (pred.score * 100).astype("float32")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred[["tid", "tag", "count"]].to_parquet(KNN_TAGS, index=False)
    pd.Series(vocab, name="tag").to_frame().to_parquet(KNN_VOCAB, index=False)
    n_test = int(is_test[targets].sum())
    print(f"[knn] kNN: {len(vocab):,} tags (df >= {min_tag_tracks}) from {has_tags.sum():,} tagged train tracks "
          f"| predicted {len(targets):,} tracks ({n_test:,} test) | {len(pred):,} rows ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- 3. evaluate on the test split
def _auc(scores: np.ndarray, pos: np.ndarray) -> float | None:
    n_pos = int(pos.sum())
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    t0 = time.time()
    tracks, tags, is_test = _load()
    vocab = pd.Index(pd.read_parquet(KNN_VOCAB).tag)
    pred = pd.read_parquet(KNN_TAGS)
    truth = tags[is_test[tags.tid.to_numpy()] & (tags["count"] >= TRUTH_MIN_COUNT) & tags.tag.isin(vocab)]
    truth_sets = truth.groupby("tid").tag.agg(set)
    pred = pred[pred.tid.isin(truth_sets.index)]
    ranked = pred.sort_values(["tid", "count"], ascending=[True, False]).groupby("tid").tag.agg(list)

    train_tags = tags[~is_test[tags.tid.to_numpy()]]
    popular = list(train_tags[train_tags.tag.isin(vocab)].groupby("tag").tid.size()
                   .sort_values(ascending=False).index[:max(ks)])

    def prf(get_list):
        out = {f"precision@{k}": 0.0 for k in ks} | {f"recall@{k}": 0.0 for k in ks}
        for tid, true in truth_sets.items():
            lst = get_list(tid)
            for k in ks:
                hit = len(true.intersection(lst[:k]))
                out[f"precision@{k}"] += hit / k
                out[f"recall@{k}"] += hit / len(true)
        return {m: round(v / len(truth_sets), 4) for m, v in out.items()}

    knn = prf(lambda tid: ranked.get(tid, []))
    base = prf(lambda tid: popular)

    # per mood tag: can predicted scores separate test tracks that have the tag from those that don't?
    ids = truth_sets.index.to_numpy()
    pos_of = {t: i for i, t in enumerate(ids)}
    aucs = {}
    for tag in MOOD_TAGS:
        if tag not in vocab:
            continue
        pos = np.zeros(len(ids), dtype=bool)
        pos[[pos_of[t] for t in truth.tid[truth.tag == tag]]] = True
        sc = np.zeros(len(ids), dtype=np.float32)
        p = pred[pred.tag == tag]
        sc[[pos_of[t] for t in p.tid]] = p["count"].to_numpy()
        a = _auc(sc, pos)
        if a is not None and pos.sum() >= 20:
            aucs[tag] = {"auc": round(a, 4), "positives": int(pos.sum())}

    n_test = int(is_test.sum())
    res = {
        "split": {"train_tracks": int((~is_test).sum()), "test_tracks": n_test,
                  "test_tracks_with_truth": int(len(truth_sets)),
                  "coverage": round(float(np.isin(ids, pred.tid.unique()).mean()), 4)},
        "truth_min_count": TRUTH_MIN_COUNT, "vocab": len(vocab),
        "knn": knn, "most_frequent_tags_baseline": base,
        "mood_tag_auc": aucs,
        "mean_mood_auc": round(float(np.mean([v["auc"] for v in aucs.values()])), 4) if aucs else None,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "eval_tags.json").write_text(json.dumps(res, indent=2))
    print(f"[knn] test tracks with truth: {len(truth_sets):,} (coverage {res['split']['coverage']:.0%})")
    for k in ks:
        print(f"[knn]   P@{k:<2} kNN {knn[f'precision@{k}']:.3f}  vs most-frequent {base[f'precision@{k}']:.3f}"
              f"   R@{k:<2} kNN {knn[f'recall@{k}']:.3f}  vs {base[f'recall@{k}']:.3f}")
    print(f"[knn]   mean mood-tag AUC {res['mean_mood_auc']} over {len(aucs)} tags -> artifacts/eval_tags.json")
    return res


def run(min_tag_tracks: int = MIN_TAG_TRACKS) -> None:
    train(min_tag_tracks=min_tag_tracks)
    evaluate()

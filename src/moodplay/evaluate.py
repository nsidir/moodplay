"""End-to-end evaluation of mood queries on the Last.fm dataset TEST split (held out from training).

1. test_tracks (main): every test-split track with held-out tags. Each track is represented only by
   what the system learned from the TRAIN split, i.e. its kNN-predicted tags, turned into index vectors
   exactly like build_index does. Probe queries ("sad songs", ...) are scored with the production
   query -> tag weights (Recommender.query_vectors). A track is relevant if its held-out Last.fm tags
   contain one of the probe's target tags. Reports ROC-AUC, precision@50 and lift over random.
2. catalogue: catalogue songs that match only test-split tracks (and have no Last.fm API tags), ranked
   by the full production score incl. popularity, vs a popularity ranking. Usually small; probes with
   fewer than MIN_POSITIVES relevant tracks are skipped.

Output: artifacts/eval_queries.json
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import musicdb as db
from .config import ARTIFACTS, SOURCE_WEIGHTS
from .knn_tags import KNN_TAGS, TRUTH_MIN_COUNT, _auc, normalize_tags

PROBES: dict[str, list[str]] = {
    "chill vibes": ["chill", "mellow", "relaxing", "downtempo"],
    "sad songs": ["sad", "melancholic", "depressing", "heartbreak"],
    "happy feel good music": ["happy", "feel good", "uplifting", "fun"],
    "energetic and upbeat": ["energetic", "upbeat"],
    "romantic love songs": ["romantic", "love songs", "sexy"],
    "dark and atmospheric": ["dark", "atmospheric", "haunting"],
    "dreamy": ["dreamy", "dream pop", "ethereal", "shoegaze"],
    "party dance music": ["dance", "party", "club"],
    "relaxing sleep music": ["relaxing", "calm", "sleep", "ambient"],
    "angry aggressive music": ["aggressive", "angry", "heavy"],
    "acoustic campfire songs": ["acoustic", "folk"],
    "summer beach": ["summer", "beach"],
    "late night driving": ["driving", "night", "late night"],
    "nostalgic oldies": ["oldies", "nostalgic", "nostalgia"],
    "groovy and funky": ["funky", "funk", "groovy"],
    "psychedelic trippy": ["psychedelic", "trippy"],
}
TOP_TEST = 50
TOP_CAT = 20
MIN_POSITIVES = 10


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _p_at(scores, pos, k):
    return float(pos[np.argsort(-scores)[:k]].mean())


def _test_truth() -> tuple[pd.DataFrame, pd.Series]:
    tracks = db.read("lastfm_tracks", ["tid", "track_id", "split"])
    test = tracks[tracks.split == "test"]
    tags = db.read("tags", ["item_id", "tag", "count"], sources=["lastfm_ds"])
    tags = tags[tags.item_id.isin(set(test.track_id)) & (tags["count"] >= TRUTH_MIN_COUNT)]
    tags["tid"] = pd.Index(test.track_id).get_indexer(tags.pop("item_id"))
    tags["tid"] = test.tid.to_numpy()[tags.tid.to_numpy()]
    truth = normalize_tags(tags).groupby("tid").tag.agg(set)
    return tracks, truth


def _vectors(rec, pred: pd.DataFrame, row_of: pd.Index, idf: pd.Series):
    """Long (tid, tag, w) predictions -> index matrices exactly like build_index (M rows L2, V = centred M.E)."""
    pred = pred[pred.tid.isin(row_of) & pred.tag.isin(rec.tag_index.keys())]
    w = pred.w.to_numpy(np.float32) * idf.reindex(pred.tag).fillna(0).to_numpy(np.float32)
    M = sp.csr_matrix((w, (row_of.get_indexer(pred.tid), pred.tag.map(rec.tag_index).to_numpy())),
                      shape=(len(row_of), len(rec.tag_names)))
    M.sum_duplicates()
    l1 = sp.diags(1.0 / np.maximum(np.asarray(M.sum(axis=1)).ravel(), 1e-9)) @ M
    V = np.asarray(l1 @ rec.E, dtype=np.float32)
    V -= V.mean(axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    return (sp.diags(1.0 / np.maximum(norms, 1e-9)) @ M).tocsr(), V


def _probes(rec, M, V, sets) -> dict:
    from .moods import plan

    per = {}
    for q, targets in PROBES.items():
        pos = np.array([bool(t & set(targets)) for t in sets])
        if pos.sum() < MIN_POSITIVES or pos.all():
            continue
        tw, qv, _ = rec.query_vectors(plan(q))
        s = rec.w.a * _z(M @ tw) + rec.w.b * _z(V @ qv)
        prev = float(pos.mean())
        p = _p_at(s, pos, TOP_TEST)
        per[q] = {"positives": int(pos.sum()), "auc": round(_auc(s, pos), 4), f"p@{TOP_TEST}": round(p, 4),
                  "prevalence": round(prev, 4), "lift": round(p / prev, 2) if prev else None}
    mean = {m: round(float(np.mean([v[m] for v in per.values()])), 4) if per else None
            for m in ("auc", f"p@{TOP_TEST}", "prevalence", "lift")}
    return {"queries": per, "mean": mean}


def _test_tracks(rec, truth: pd.Series) -> dict:
    """Represent held-out tracks only by what was learned from the train split, rank them for each probe."""
    from .audio_tags import AUDIO_TAGS_LASTFM

    idf = pd.read_parquet(ARTIFACTS / "tags.parquet").set_index("tag").idf
    sw = SOURCE_WEIGHTS
    knn = pd.read_parquet(KNN_TAGS).assign(w=lambda d: d["count"] / 100.0 * sw["lastfm_knn"])
    row_of = pd.Index(truth.index.to_numpy())
    M, V = _vectors(rec, knn, row_of, idf)
    out = {"tracks": int(len(row_of)), "with_predictions": int(knn[knn.tid.isin(row_of)].tid.nunique()),
           **_probes(rec, M, V, truth.to_numpy())}
    if AUDIO_TAGS_LASTFM.exists():
        audio = pd.read_parquet(AUDIO_TAGS_LASTFM).assign(w=lambda d: d["count"] / 100.0 * sw["audio"])
        sub = pd.Index(np.intersect1d(row_of, audio.tid.unique()))
        sets = truth.loc[sub].to_numpy()
        variants = {"knn": knn, "audio": audio, "knn+audio": pd.concat([knn, audio], ignore_index=True)}
        out["with_audio"] = {"tracks": int(len(sub))}
        for name, pred in variants.items():
            Ms, Vs = _vectors(rec, pred, sub, idf)
            out["with_audio"][name] = _probes(rec, Ms, Vs, sets)
    return out


def _catalogue(rec, tracks: pd.DataFrame, truth: pd.Series) -> dict:
    match_path = ARTIFACTS / "lastfm_match.parquet"
    if not match_path.exists():
        return {"population": 0, "queries": {}, "note": "no artifacts/lastfm_match.parquet"}
    match = pd.read_parquet(match_path)
    g = match.groupby("recording_mbid").agg(test_only=("split", lambda s: (s == "test").all()),
                                            has_api=("has_api", "any"))
    keep = set(g.index[g.test_only & ~g.has_api]) & set(rec.row_of)
    match = match[match.recording_mbid.isin(keep) & match.tid.isin(truth.index)]
    cat_truth = match.groupby("recording_mbid").tid.agg(lambda t: set().union(*truth.loc[list(t)]))
    if cat_truth.empty:
        return {"population": 0, "queries": {}}
    cand = np.array([rec.row_of[m] for m in cat_truth.index])
    sets = cat_truth.to_numpy()
    pop = rec.meta.users.to_numpy(dtype=float)[cand] + np.random.default_rng(0).random(len(cand)) * 1e-3
    per, skipped = {}, []
    for q, targets in PROBES.items():
        pos = np.array([bool(t & set(targets)) for t in sets])
        if pos.sum() < MIN_POSITIVES or pos.all():
            skipped.append(q)
            continue
        s = rec.score_query(q, cand)
        per[q] = {"positives": int(pos.sum()), "auc": round(_auc(s, pos), 4), "auc_popularity": round(_auc(pop, pos), 4),
                  f"p@{TOP_CAT}": round(_p_at(s, pos, TOP_CAT), 4),
                  f"p@{TOP_CAT}_popularity": round(_p_at(pop, pos, TOP_CAT), 4), "prevalence": round(float(pos.mean()), 4)}
    return {"population": int(len(cand)), "queries": per, "skipped_queries": skipped}


def run() -> dict:
    t0 = time.time()
    db.open_db()
    if not KNN_TAGS.exists():
        raise SystemExit("No artifacts/knn_tags.parquet. Run `make knn build` first.")
    from .recommend import Recommender

    rec = Recommender()
    tracks, truth = _test_truth()
    res = {"truth_min_count": TRUTH_MIN_COUNT, "test_tracks": _test_tracks(rec, truth),
           "catalogue": _catalogue(rec, tracks, truth),
           "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0)}
    (ARTIFACTS / "eval_queries.json").write_text(json.dumps(res, indent=2))

    tt = res["test_tracks"]
    print(f"[eval] held-out test tracks: {tt['tracks']:,} ({tt['with_predictions']:,} with kNN tags)")
    print(f"[eval] {'query':26s} {'pos':>6s} {'AUC':>6s} {'P@' + str(TOP_TEST):>6s} {'rand':>6s} {'lift':>5s}")
    for q, v in tt["queries"].items():
        print(f"[eval] {q:26s} {v['positives']:6d} {v['auc']:6.3f} {v[f'p@{TOP_TEST}']:6.3f} "
              f"{v['prevalence']:6.3f} {v['lift']:5.1f}")
    m = tt["mean"]
    if m["auc"] is not None:
        print(f"[eval] {'mean':26s} {'':6s} {m['auc']:6.3f} {m[f'p@{TOP_TEST}']:6.3f} {m['prevalence']:6.3f} {m['lift']:5.1f}")
    wa = tt.get("with_audio")
    if wa:
        print(f"[eval] same {wa['tracks']:,} test tracks that also have audio:   mean AUC / P@{TOP_TEST} / lift")
        for name in ("knn", "audio", "knn+audio"):
            m = wa[name]["mean"]
            if m["auc"] is not None:
                print(f"[eval]   {name:10s} {m['auc']:.3f} / {m[f'p@{TOP_TEST}']:.3f} / {m['lift']:.1f}")
    c = res["catalogue"]
    print(f"[eval] catalogue test-only songs: {c['population']:,}; {len(c['queries'])} probes with >= "
          f"{MIN_POSITIVES} positives -> artifacts/eval_queries.json")
    for q, v in c["queries"].items():
        print(f"[eval]   {q:26s} AUC {v['auc']:.3f} (popularity {v['auc_popularity']:.3f})")
    return res

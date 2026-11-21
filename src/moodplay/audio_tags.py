"""Audio -> mood-tag model: AcousticBrainz features, Last.fm train-split tags as labels.

Most catalogue songs have no Last.fm crowd tags, but many have AcousticBrainz audio analysis. This model
learns which audio profile goes with which crowd tag and predicts tags for every recording with audio.

data      features per recording MBID (mean over submissions): the AB high-level classifier probabilities
          (moods, genres, danceability, timbre, voice, ...) + low-level rhythm / loudness / tonal values.
          Recordings are linked to the Last.fm dataset by song_key (normalized artist|title), taken from
          musicdb.recordings when the MBID is there, else from the audio file's own artist/title tags.
          Rows are aggregated per song_key; a song is TEST if all its Last.fm tracks are in the test split.
train     multi-label MLP (PyTorch, CPU) on TRAIN songs, early stopping on a slice of TRAIN.
evaluate  on TEST songs vs their held-out tags: P@k / R@k, ROC-AUC per mood tag; compared on the same songs with
          the kNN tag model and with both combined -> artifacts/eval_audio.json
predict   tags for every musicdb.recordings MBID with audio -> artifacts/audio_tags.parquet (source "audio" in
          build_data), and for Last.fm tracks -> artifacts/audio_tags_lastfm.parquet (used by `make evaluate`).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

from . import musicdb as db
from .config import ARTIFACTS, song_key
from .knn_tags import KNN_TAGS, MOOD_TAGS, TRUTH_MIN_COUNT, _auc, normalize_tags

AUDIO_TAGS = ARTIFACTS / "audio_tags.parquet"
AUDIO_TAGS_LASTFM = ARTIFACTS / "audio_tags_lastfm.parquet"
AUDIO_FEATURES = ARTIFACTS / "audio_features.parquet"
HL_COLS = [db.ab_col(c, k) for c, ks in db.AB_CLASSIFIERS.items() for k in ks]
LL_COLS = {"ab_rhythm": ["bpm", "danceability", "onset_rate", "bpm_histogram_first_peak_bpm_mean",
                         "bpm_histogram_second_peak_bpm_mean"],
           "ab_lowlevel": ["average_loudness", "dynamic_complexity", "mfcc_zero_mean"],
           "ab_tonal": ["key_scale", "tuning_frequency"]}
BATCH_ROWS = 200_000
MIN_POS = 100       # a tag needs this many positive TRAIN songs to be learned
MAX_TAGS = 400      # most frequent tags learned (+ every query-lexicon tag with >= MIN_POS); bounds memory
TOP_PRED = 20
MIN_PROB = 0.05


# ------------------------------------------------------------------ features
def _agg(df: pd.DataFrame, cols: list[str], by: str = "key") -> pd.DataFrame:
    """Sum + count per key (so partial aggregates from several parts can be merged)."""
    g = df.groupby(by, sort=False)[cols]
    out = g.sum(min_count=1)
    out["_n"] = g.size().astype("int32")
    return out


def _merge(run: pd.DataFrame | None, agg: pd.DataFrame) -> pd.DataFrame:
    return agg if run is None else pd.concat([run, agg]).groupby(level=0, sort=False).sum(min_count=1)


def build_features() -> pd.DataFrame:
    """Mean audio features per row key: the recording MBID for MBIDs in musicdb.recordings (they need their own
    predictions), else 'sk:<song_key>' for other submissions that match a Last.fm dataset song (training only).
    Aggregates are merged part by part, so memory stays bounded by the number of keys, not submissions."""
    t0 = time.time()
    db.open_db()
    if not db.has("ab_highlevel"):
        raise SystemExit("musicdb has no ab_highlevel yet. Run `make fetch db` first.")
    rec = db.read("recordings", ["recording_mbid", "song_key"]).dropna()
    rec_key = dict(zip(rec.recording_mbid, rec.song_key))
    lf_keys = set(db.read("lastfm_tracks", ["song_key"]).song_key.dropna())

    hl, maps = None, []
    ds_hl = db.dataset("ab_highlevel")
    for part in db.partitions("ab_highlevel"):
        kept = seen = 0
        # 200k-row batches keep peak memory well under 1 GB per part
        for batch in ds_hl.to_batches(columns=["mbid", "tag_artist", "tag_title", *HL_COLS],
                                      filter=pads.field("part") == part, batch_size=BATCH_ROWS):
            t = batch.to_pandas()
            in_rec = t.mbid.isin(rec_key.keys()).to_numpy()
            # song_key from the audio file tags for MBIDs we have no metadata for
            sk = np.array([song_key(a, ti) if a and ti and not r else ""
                           for a, ti, r in zip(t.tag_artist, t.tag_title, in_rec)], dtype=object)
            keep = in_rec | pd.Series(sk).isin(lf_keys).to_numpy()
            key = np.where(in_rec, t.mbid.to_numpy(dtype=object), "sk:" + sk)
            t = t.loc[keep, ["mbid", *HL_COLS]].assign(key=key[keep])
            hl = _merge(hl, _agg(t, HL_COLS))
            maps.append(t[["mbid", "key"]].drop_duplicates().astype("string[pyarrow]"))
            kept, seen = kept + int(keep.sum()), seen + len(keep)
            del t, batch
        print(f"[audio] part {part}: kept {kept:,}/{seen:,} submissions, {len(hl):,} keys "
              f"({time.time() - t0:.0f}s)")
    n = hl.pop("_n")
    feats = hl.div(n, axis=0).astype("float32")
    del hl
    mbid_key = pd.concat(maps, ignore_index=True).drop_duplicates("mbid")
    del maps

    wanted = pa.array(mbid_key.mbid.to_numpy(dtype=object), type=pa.string())
    for table, cols in LL_COLS.items():
        if not db.has(table):
            continue
        a = None
        for batch in db.dataset(table).to_batches(columns=["mbid", *cols], filter=pads.field("mbid").isin(wanted),
                                                  batch_size=BATCH_ROWS):
            t = batch.to_pandas()
            if "key_scale" in t:
                t["key_scale"] = (t.key_scale == "major").astype("float32")
            t = t.astype({"mbid": "string[pyarrow]"}).merge(mbid_key, on="mbid")
            a = _merge(a, _agg(t, cols))
            del t, batch
        if a is None:
            continue
        n = a.pop("_n")
        feats = feats.join(a.div(n, axis=0).astype("float32"), how="left")
        print(f"[audio] {table}: {len(a):,} keys ({time.time() - t0:.0f}s)")
    idx = feats.index.to_series()
    feats["in_recordings"] = ~idx.str.startswith("sk:").to_numpy()
    feats["song_key"] = np.where(feats.in_recordings, idx.map(rec_key), idx.str[3:])
    feats.index.name = "mbid"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    feats.reset_index().to_parquet(AUDIO_FEATURES, index=False)
    print(f"[audio] features: {len(feats):,} keys x {len(feats.columns) - 2} "
          f"({feats.in_recordings.sum():,} musicdb recordings, {(~feats.in_recordings).sum():,} other Last.fm songs) "
          f"-> {AUDIO_FEATURES.name} ({time.time() - t0:.0f}s)")
    return feats


# ------------------------------------------------------------------ labels
def _song_labels():
    tracks = db.read("lastfm_tracks", ["tid", "track_id", "split", "song_key"]).sort_values("tid")
    split = tracks.groupby("song_key").split.agg(lambda s: "test" if (s == "test").all() else "train")
    tags = db.read("tags", ["item_id", "tag", "count"], sources=["lastfm_ds"])
    tags["tid"] = tracks.tid.to_numpy()[pd.Index(tracks.track_id).get_indexer(tags.pop("item_id"))]
    tags = normalize_tags(tags[tags["count"] >= TRUTH_MIN_COUNT])
    tags["song_key"] = tracks.song_key.to_numpy()[tags.tid.to_numpy()]
    return tracks, split, tags.groupby(["song_key", "tag"], as_index=False)["count"].max()


def _matrix(feats: pd.DataFrame, cols: list[str], mu=None, sd=None):
    X = feats[cols].to_numpy(np.float32)
    if mu is None:
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0) + 1e-6
    X = (X - mu) / sd
    return np.nan_to_num(X, nan=0.0).astype(np.float32), mu, sd


def _vocab(train_df: pd.Series) -> list[str]:
    """Tags worth learning: the MAX_TAGS most frequent among TRAIN songs, plus every tag the query lexicon
    (moods.py) can ask for, as long as it has MIN_POS positive train songs."""
    from .moods import FEELINGS, VIBES

    lexicon = {t for v in VIBES.values() for t in v} | {t for x in FEELINGS.values() for lst in x for t in lst}
    ok = train_df[train_df >= MIN_POS]
    return sorted(set(ok.sort_values(ascending=False).index[:MAX_TAGS]) | (set(ok.index) & lexicon))


def _train_mlp(X, Y, Xv, Yv, epochs: int = 40, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(512, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)          # labels stay uint8, cast per minibatch
    Xvt, Yvt = torch.from_numpy(Xv), torch.from_numpy(Yv).float()
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            loss_fn(net(Xt[b]), Yt[b].float()).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xvt), Yvt))
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    print(f"[audio] MLP stopped after {ep + 1} epochs, best validation loss {best:.4f}")
    return net


def _predict(net, X) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + 65536]))).numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def _predict_long(net, X, ids, vocab: list[str], idname: str, chunk: int = 50_000) -> pd.DataFrame:
    """Predict + keep top tags in chunks, so the full (rows x tags) matrix is never in memory."""
    ids = np.asarray(ids)
    return pd.concat([_long(ids[i:i + chunk], _predict(net, X[i:i + chunk]), vocab, idname)
                      for i in range(0, len(X), chunk)], ignore_index=True)


def _long(ids, P: np.ndarray, vocab: list[str], idname: str) -> pd.DataFrame:
    """Top TOP_PRED tags per row with prob >= MIN_PROB, as (id, tag, count 0-100)."""
    k = min(TOP_PRED, P.shape[1])
    top = np.argsort(-P, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(P)), k)
    cols = top.ravel()
    p = P[rows, cols]
    keep = p >= MIN_PROB
    return pd.DataFrame({idname: np.asarray(ids)[rows[keep]], "tag": np.asarray(vocab)[cols[keep]],
                         "count": (p[keep] * 100).astype(np.float32)})


# ------------------------------------------------------------------ train + evaluate + predict
def run(rebuild_features: bool = False, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    t0 = time.time()
    db.open_db()
    feats = pd.read_parquet(AUDIO_FEATURES).set_index("mbid") if AUDIO_FEATURES.exists() and not rebuild_features \
        else build_features()
    cols = [c for c in feats.columns if c not in ("song_key", "in_recordings")]
    tracks, split, labels = _song_labels()

    # one row per labelled song (mean over its recordings)
    lab = feats[feats.song_key.isin(split.index)]
    songs = lab.groupby("song_key")[cols].mean()
    songs["split"] = split.reindex(songs.index).to_numpy()
    pos = labels[labels.song_key.isin(songs.index)]
    train_keys = songs.index[songs.split == "train"]
    df = pos[pos.song_key.isin(train_keys)].groupby("tag").song_key.nunique()
    vocab = _vocab(df)
    vi = {t: i for i, t in enumerate(vocab)}
    row = pd.Series(np.arange(len(songs)), index=songs.index)
    Y = np.zeros((len(songs), len(vocab)), np.uint8)  # bytes, not floats: 4x less memory
    p2 = pos[pos.tag.isin(vi)]
    Y[row[p2.song_key].to_numpy(), p2.tag.map(vi).to_numpy()] = 1

    is_tr = (songs.split == "train").to_numpy()
    tr_idx = np.flatnonzero(is_tr)
    rng = np.random.default_rng(0)
    rng.shuffle(tr_idx)
    n_val = max(len(tr_idx) // 10, 1)
    val_idx, fit_idx = tr_idx[:n_val], tr_idx[n_val:]
    te_idx = np.flatnonzero(~is_tr & (Y.sum(axis=1) > 0))
    X, mu, sd = _matrix(songs.iloc[fit_idx], cols)
    Xall, _, _ = _matrix(songs, cols, mu, sd)
    print(f"[audio] songs with audio + Last.fm tags: {len(songs):,} (train {len(fit_idx):,}, val {len(val_idx):,}, "
          f"test {len(te_idx):,}) | {len(vocab)} tags (top {MAX_TAGS} + query-lexicon tags, >= {MIN_POS} train songs)")
    net = _train_mlp(Xall[fit_idx], Y[fit_idx], Xall[val_idx], Y[val_idx])

    # ---- evaluate on TEST songs; compare with kNN on the same songs ----
    P_te = _predict(net, Xall[te_idx])
    te_keys = songs.index[te_idx]
    knn = pd.read_parquet(KNN_TAGS) if KNN_TAGS.exists() else None
    K = np.zeros_like(P_te)
    if knn is not None:
        tk = tracks[tracks.song_key.isin(te_keys)][["tid", "song_key"]]
        kk = knn.merge(tk, on="tid")
        kk = kk[kk.tag.isin(vi)].groupby(["song_key", "tag"], as_index=False)["count"].max()
        tr = pd.Series(np.arange(len(te_keys)), index=te_keys)
        K[tr[kk.song_key].to_numpy(), kk.tag.map(vi).to_numpy()] = kk["count"].to_numpy() / 100.0
    Yte = Y[te_idx]
    has_knn = K.sum(axis=1) > 0

    def scores(S, mask):
        res = {}
        order = np.argsort(-S[mask], axis=1)
        Ym = Yte[mask]
        for k in ks:
            hits = np.take_along_axis(Ym, order[:, :k], axis=1).sum(axis=1)
            res[f"precision@{k}"] = round(float((hits / k).mean()), 4)
            res[f"recall@{k}"] = round(float((hits / np.maximum(Ym.sum(axis=1), 1)).mean()), 4)
        aucs = {}
        for t in MOOD_TAGS:
            if t in vi and Ym[:, vi[t]].sum() >= 20:
                aucs[t] = round(_auc(S[mask][:, vi[t]], Ym[:, vi[t]].astype(bool)), 4)
        res["mood_tag_auc"] = aucs
        res["mean_mood_auc"] = round(float(np.mean(list(aucs.values()))), 4) if aucs else None
        return res

    both = has_knn
    combo = 0.5 * P_te + 0.5 * K
    result = {
        "songs": {"train": int(len(fit_idx)), "val": int(len(val_idx)), "test": int(len(te_idx)),
                  "test_with_knn": int(both.sum())}, "vocab": len(vocab), "features": len(cols),
        "audio_all_test": scores(P_te, np.ones(len(te_idx), bool)),
        "same_test_songs": ({"audio": scores(P_te, both), "knn": scores(K, both), "audio+knn": scores(combo, both)}
                            if both.any() else None),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---- predict: every musicdb recording with audio, and every Last.fm track's song ----
    rec_f = feats[feats.in_recordings]
    Xr, _, _ = _matrix(rec_f, cols, mu, sd)
    _predict_long(net, Xr, rec_f.index, vocab, "recording_mbid").to_parquet(AUDIO_TAGS, index=False)
    lf = songs.index.to_series()
    tl = tracks[tracks.song_key.isin(lf.index)][["tid", "song_key"]]
    song_pred = _predict_long(net, Xall, songs.index, vocab, "song_key")
    song_pred.merge(tl, on="song_key")[["tid", "tag", "count"]].to_parquet(AUDIO_TAGS_LASTFM, index=False)
    result["predicted_recordings"] = int(len(rec_f))
    result["seconds"] = round(time.time() - t0)
    (ARTIFACTS / "eval_audio.json").write_text(json.dumps(result, indent=2))

    a, s = result["audio_all_test"], result["same_test_songs"]
    pk = " ".join(f"P@{k} {a[f'precision@{k}']:.3f}" for k in ks)
    print(f"[audio] TEST songs: {pk}  R@{ks[-1]} {a[f'recall@{ks[-1]}']:.3f}  mean mood AUC {a['mean_mood_auc']}")
    k = ks[len(ks) // 2]
    if s:
        print(f"[audio] same {int(both.sum()):,} test songs   mood AUC  audio {s['audio']['mean_mood_auc']}  "
              f"kNN {s['knn']['mean_mood_auc']}  audio+kNN {s['audio+knn']['mean_mood_auc']}   P@{k}  audio "
              f"{s['audio'][f'precision@{k}']}  kNN {s['knn'][f'precision@{k}']}  "
              f"audio+kNN {s['audio+knn'][f'precision@{k}']}")
    print(f"[audio] predicted tags for {len(rec_f):,} musicdb recordings -> {AUDIO_TAGS.name} ({result['seconds']}s)")
    return result

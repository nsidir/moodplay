"""Offline step: embed tags, build track vectors.

  E (T x 384)  tag embeddings ("{tag} music"), L2-normalized
  M (N x T)    sparse tag weights per track, rows L2-normalized       -> tag matching (A)
  V (N x 384)  (L1-rows M) @ E, mean-centered, L2-normalized          -> embedding match (B)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import ARTIFACTS, EMBED_MODEL, TAG_TEMPLATE


def run() -> None:
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    tracks = pd.read_parquet(ARTIFACTS / "tracks.parquet")
    tags = pd.read_parquet(ARTIFACTS / "tags.parquet").reset_index(drop=True)
    tt = pd.read_parquet(ARTIFACTS / "track_tags.parquet")

    row_of = pd.Series(np.arange(len(tracks)), index=tracks.recording_mbid)
    col_of = pd.Series(np.arange(len(tags)), index=tags.tag)
    tt = tt[tt.recording_mbid.isin(row_of.index) & tt.tag.isin(col_of.index)]
    M = sp.csr_matrix((tt.weight.to_numpy(np.float32), (row_of[tt.recording_mbid].to_numpy(),
                       col_of[tt.tag].to_numpy())), shape=(len(tracks), len(tags)))

    print(f"[index] embedding {len(tags):,} tags with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    E = model.encode([TAG_TEMPLATE.format(tag=t) for t in tags.tag], normalize_embeddings=True,
                     batch_size=128, show_progress_bar=False).astype(np.float32)

    l1 = sp.diags(1.0 / np.maximum(np.asarray(M.sum(axis=1)).ravel(), 1e-9)) @ M
    V = np.asarray(l1 @ E, dtype=np.float32)
    V -= V.mean(axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9

    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    M2 = (sp.diags(1.0 / np.maximum(norms, 1e-9)) @ M).tocsr().astype(np.float32)

    np.savez_compressed(ARTIFACTS / "index.npz", track_mbids=tracks.recording_mbid.to_numpy().astype(str),
                        tag_names=tags.tag.to_numpy().astype(str), E=E, V=V)
    sp.save_npz(ARTIFACTS / "M.npz", M2)
    print(f"[index] {len(tracks):,} tracks x {len(tags):,} tags ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    run()

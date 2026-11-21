"""Online: natural-language vibe -> ranked, diversified tracks.

score = a*z(tag match) + b*z(embedding match) + pop*z(log listeners) - exclude*penalty
        + personal * conf(n) * z(implicit-ALS preference)
then MMR over V with a per-artist cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .als import ALS_FILE, ALS_META, fold_in
from .config import ARTIFACTS, EMBED_MODEL
from .moods import FEELINGS, VIBES, QueryPlan, lexicon_tags, plan


@dataclass
class Weights:
    a: float = 0.55
    b: float = 0.45
    exclude: float = 0.9
    popularity: float = 0.25
    emb_top_tags: int = 12
    emb_temp: float = 0.05
    lex_share: float = 0.75
    feeling_weight: float = 1.5
    personal: float = 0.8
    personal_half: float = 4.0
    mmr_lambda: float = 0.7
    pool: int = 300
    per_artist: int = 2


@dataclass
class Track:
    mbid: str
    title: str
    artist: str
    release: str | None
    year: int | None
    users: int
    score: float
    why: list[str] = field(default_factory=list)
    cover: str | None = None
    personal: float | None = None
    because: str | None = None

    @property
    def search(self) -> str:
        return quote_plus(f"{self.artist} {self.title}")

    @property
    def links(self) -> dict[str, str]:
        return {
            "Spotify": f"https://open.spotify.com/search/{self.search}",
            "YouTube": f"https://www.youtube.com/results?search_query={self.search}",
            "Last.fm": f"https://www.last.fm/search/tracks?q={self.search}",
            "MusicBrainz": f"https://musicbrainz.org/recording/{self.mbid}",
            "ListenBrainz": f"https://listenbrainz.org/player/?recording_mbids={self.mbid}",
        }


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-9)


class Recommender:
    def __init__(self, weights: Weights | None = None):
        from sentence_transformers import SentenceTransformer

        self.w = weights or Weights()
        idx = np.load(ARTIFACTS / "index.npz", allow_pickle=False)
        self.mbids, self.tag_names = idx["track_mbids"], idx["tag_names"]
        self.E, self.V = idx["E"], idx["V"]
        self.M = sp.load_npz(ARTIFACTS / "M.npz").tocsr()
        self.MT = self.M.T.tocsr()
        self.tag_index = {str(t): i for i, t in enumerate(self.tag_names)}
        self.row_of = {str(m): i for i, m in enumerate(self.mbids)}
        meta = pd.read_parquet(ARTIFACTS / "tracks.parquet").set_index("recording_mbid").loc[self.mbids]
        self.meta = meta.reset_index()
        self._pop = _z(np.log1p(self.meta.users.to_numpy(dtype=float)))
        self._artist = self.meta.artist.fillna("").str.lower().to_numpy()
        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.reload_als()

    def reload_als(self) -> None:
        self.Y = np.load(ALS_FILE) if ALS_FILE.exists() else None
        self.als_meta = json.loads(ALS_META.read_text()) if ALS_META.exists() else {}
        if self.Y is not None:
            self.YtY = self.Y.T @ self.Y
            self._Yn = self.Y / np.maximum(np.linalg.norm(self.Y, axis=1, keepdims=True), 1e-9)

    def missing_lexicon_tags(self) -> set[str]:
        tags = {t for v in VIBES.values() for t in v} | {t for x in FEELINGS.values() for lst in x for t in lst}
        return tags - set(self.tag_index)

    # ---- query -> weights ----
    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _emb_tags(self, q: np.ndarray, k: int | None = None) -> np.ndarray:
        sims = self.E @ q
        top = np.argsort(-sims)[: k or self.w.emb_top_tags]
        out = np.zeros(len(self.tag_names), dtype=np.float32)
        e = np.exp((sims[top] - sims[top].max()) / self.w.emb_temp)
        out[top] = e / e.sum()
        return out

    def _lex_vec(self, tags) -> np.ndarray:
        out = np.zeros(len(self.tag_names), dtype=np.float32)
        for t, wt in (tags.items() if isinstance(tags, dict) else ((t, 1.0) for t in tags)):
            if t in self.tag_index:
                out[self.tag_index[t]] += wt
        s = out.sum()
        return out / s if s > 0 else out

    def query_vectors(self, p: QueryPlan):
        tw = np.zeros(len(self.tag_names), dtype=np.float32)
        aspects = p.aspects or ([] if p.feeling_tags else [(p.text, [])])
        embs = self._encode([a for a, _ in aspects]) if aspects else np.zeros((0, self.E.shape[1]), np.float32)
        for (_, lex), q in zip(aspects, embs):
            d, lv = self._emb_tags(q), self._lex_vec(lex)
            tw += self.w.lex_share * lv + (1 - self.w.lex_share) * d if lv.any() else d
        if p.feeling_tags:
            fv = self._lex_vec(p.feeling_tags)
            tw += self.w.feeling_weight * (fv if fv.any() else self._emb_tags(self._encode([" ".join(p.feeling_tags)])[0]))
        tw /= tw.sum() + 1e-9
        centroid = _unit(self.E.T @ tw)
        qv = _unit(0.5 * _unit(embs.mean(axis=0)) + 0.5 * centroid) if len(embs) else centroid

        ex = self._lex_vec(p.exclude_tags)
        free = [e for e in p.excludes if not lexicon_tags(e)]
        for q in (self._encode(free) if free else []):
            ex += self._emb_tags(q, 4)
        return tw, qv.astype(np.float32), ex

    # ---- personalization ----
    def user_vector(self, feedback: dict[str, int]):
        if self.Y is None:
            return None, []
        rows = [self.row_of[m] for m in feedback if m in self.row_of and self.Y[self.row_of[m]].any()]
        if not rows:
            return None, []
        prefs = [feedback[str(self.mbids[r])] for r in rows]
        u = fold_in(self.Y, self.YtY, rows, prefs, self.als_meta.get("reg", 0.1), self.als_meta.get("alpha", 20.0))
        return u, rows

    # ---- ranking ----
    def recommend(self, text: str, k: int = 15, min_users: int = 20, years: tuple[int, int] | None = None,
                  mood_mode: str = "lift", feedback: dict[str, int] | None = None,
                  hide_rated: bool = True) -> tuple[list[Track], dict]:
        p = plan(text, mood_mode)
        tw, qv, ex = self.query_vectors(p)
        score_a = self.M @ tw
        score_b = self.V @ qv
        penalty = self.M @ ex if ex.any() else None

        mask = self.meta.users.to_numpy() >= min_users
        if years:
            y = self.meta.year.to_numpy(dtype=float)
            mask &= (y >= years[0]) & (y <= years[1])
        feedback = feedback or {}
        if hide_rated:
            for m in feedback:
                if m in self.row_of:
                    mask[self.row_of[m]] = False
        cand = np.flatnonzero(mask)
        u, rated = self.user_vector(feedback)

        top_t = np.argsort(-tw)[:8]
        info = {"feeling": p.feeling, "mood_mode": mood_mode, "excludes": p.excludes,
                "query_tags": [(str(self.tag_names[i]), float(tw[i])) for i in top_t if tw[i] > 0],
                "personalized": u is not None, "n_feedback": len(feedback)}
        if len(cand) == 0:
            return [], info

        w = self.w
        s = w.a * _z(score_a[cand]) + w.b * _z(score_b[cand]) + w.popularity * self._pop[cand]
        if penalty is not None:
            s -= w.exclude * np.maximum(_z(penalty[cand]), 0)
        pred = None
        if u is not None:
            pred = self.Y @ u
            conf = len(rated) / (len(rated) + w.personal_half)
            has = self.Y[cand].any(axis=1)
            s += w.personal * conf * np.where(has, _z(pred[cand]), 0.0)
        liked = [r for r in rated if feedback[str(self.mbids[r])] > 0]

        out = []
        for i, sc in self._mmr(cand, s, k):
            m = self.meta.iloc[i]
            row = self.M.getrow(i)
            why = sorted(zip(row.indices, row.data), key=lambda t: -(tw[t[0]] + 1e-3) * t[1])[:4]
            because = None
            if liked and pred is not None and self.Y[i].any():
                sims = self._Yn[liked] @ self._Yn[i]
                j = int(np.argmax(sims))
                if sims[j] > 0.5:
                    because = f"{self.meta.title.iat[liked[j]]} - {self.meta.artist.iat[liked[j]]}"
            cover = (f"https://coverartarchive.org/release/{m.caa_release_mbid}/{int(m.caa_id)}-250.jpg"
                     if pd.notna(m.caa_id) and pd.notna(m.caa_release_mbid) else None)
            out.append(Track(str(m.recording_mbid), m.title, m.artist, m.release,
                             None if pd.isna(m.year) else int(m.year), int(m.users), float(sc),
                             [str(self.tag_names[j]) for j, _ in why], cover,
                             None if pred is None or not self.Y[i].any() else float(pred[i]), because))
        return out, info

    def _mmr(self, cand: np.ndarray, s: np.ndarray, k: int) -> list[tuple[int, float]]:
        order = np.argsort(-s)[: self.w.pool]
        idx, raw = cand[order], s[order]
        rel = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        Vc = self.V[idx]
        lam = self.w.mmr_lambda
        chosen: list[int] = []
        per_artist: dict[str, int] = {}
        max_sim = np.zeros(len(idx), dtype=np.float32)
        blocked = np.zeros(len(idx), dtype=bool)
        while len(chosen) < min(k, len(idx)):
            mmr = lam * rel - (1 - lam) * max_sim if chosen else rel.copy()
            mmr[blocked] = -np.inf
            j = int(np.argmax(mmr))
            if not np.isfinite(mmr[j]):
                break
            chosen.append(j)
            blocked[j] = True
            a = self._artist[idx[j]]
            per_artist[a] = per_artist.get(a, 0) + 1
            if per_artist[a] >= self.w.per_artist:
                blocked |= self._artist[idx] == a
            max_sim = np.maximum(max_sim, np.maximum(Vc @ Vc[j], 0))
        return [(int(idx[j]), float(raw[j])) for j in chosen]

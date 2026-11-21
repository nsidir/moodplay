"""music-vibe: recommend tracks from a natural-language mood.

  uv run music-vibe extract                  # download + stream ListenBrainz sample -> data/lb
  uv run music-vibe crawl --tracks 20000     # Last.fm mood tags (needs LASTFM_API_KEY in .env)
  uv run music-vibe build                    # Spark data step + embedding index
  uv run music-vibe train-als                # Spark implicit ALS
  uv run music-vibe ask "chill late night"   # one-shot query
  uv run music-vibe                          # interactive mode
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")


def _years(s: str) -> tuple[int, int]:
    lo, _, hi = s.partition("-")
    return int(lo or 0), int(hi or 9999)


def _print(recs, info, explain: bool = False) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if info.get("feeling"):
        print(f"  (feeling: {info['feeling']} - mode: {info['mood_mode']})")
    if info["excludes"]:
        print(f"  (excluding: {', '.join(info['excludes'])})")
    if explain:
        print("  query tags: " + ", ".join(f"{t} {w:.2f}" for t, w in info["query_tags"][:6]))
    if not recs:
        print("  no matches - try relaxing filters (--min-users, --years)")
    for n, r in enumerate(recs, 1):
        yr = f" ({r.year})" if r.year else ""
        print(f"{n:>3}. {r.artist} - {r.title}{yr}")
        print(f"       vibe: {', '.join(r.why)}")


def main() -> None:
    p = argparse.ArgumentParser(prog="music-vibe", description="Music recommendations from your mood.")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("extract", help="download the ListenBrainz sample dump and extract parquet")

    c = sub.add_parser("crawl", help="fetch Last.fm mood tags (cached, resumable)")
    c.add_argument("--seed-pages", type=int, default=5, help="pages of 100 per seed tag (0 to skip)")
    c.add_argument("--tracks", type=int, default=20000, help="top-N catalogue tracks to tag (0 to skip)")
    c.add_argument("--rate", type=float, default=4.0, help="requests per second (keep <= 5)")

    def data_args(x):
        x.add_argument("--catalog-size", type=int, default=60000)
        x.add_argument("--top-tags", type=int, default=25)
        x.add_argument("--min-tag-tracks", type=int, default=15)
        x.add_argument("--driver-memory", default="3g")

    b = sub.add_parser("build", help="build-data + build-index")
    b.add_argument("--skip-spark", action="store_true")
    data_args(b)
    data_args(sub.add_parser("build-data", help="Spark: catalogue + weighted tags -> artifacts/"))
    sub.add_parser("build-index", help="tag embeddings + track vectors -> artifacts/index.npz")

    t = sub.add_parser("train-als", help="Spark implicit ALS on ListenBrainz feedback (+ app feedback)")
    t.add_argument("--rank", type=int, default=48)
    t.add_argument("--iters", type=int, default=15)
    t.add_argument("--reg", type=float, default=0.1)
    t.add_argument("--alpha", type=float, default=20.0, help="confidence weight for feedback")
    t.add_argument("--holdout", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--driver-memory", default="3g")
    t.add_argument("--feedback", help="parquet with user_id, recording_mbid, value (+1/-1)")

    for name in ("ask", "chat"):
        a = sub.add_parser(name, help="one-shot query" if name == "ask" else "interactive mode")
        if name == "ask":
            a.add_argument("query", nargs="+")
        a.add_argument("-k", type=int, default=15)
        a.add_argument("--min-users", type=int, default=20, help="minimum ListenBrainz listeners")
        a.add_argument("--years", type=_years, help="e.g. 1990-2005, 2000-, -1980")
        a.add_argument("--mood", choices=["lift", "match"], default="lift")
        a.add_argument("--explain", action="store_true")
    args = p.parse_args(sys.argv[1:] or ["chat"])

    if args.cmd == "extract":
        from . import extract
        extract.run()
        return
    if args.cmd == "crawl":
        from .lastfm import LastFM
        lf = LastFM(rate=args.rate)
        if args.seed_pages:
            lf.seed_pass(pages=args.seed_pages)
        if args.tracks:
            import pandas as pd
            from .config import ARTIFACTS
            path = ARTIFACTS / "tracks.parquet"
            if not path.exists():
                sys.exit("Run `music-vibe build` once first so the crawler knows the catalogue.")
            tr = pd.read_parquet(path, columns=["recording_mbid", "artist", "title", "users"])
            tr = tr.sort_values("users", ascending=False)
            lf.track_pass(list(tr[["recording_mbid", "artist", "title"]].itertuples(index=False, name=None)),
                          limit=args.tracks)
        return
    if args.cmd in ("build", "build-data", "build-index"):
        from . import build_data, build_index
        if args.cmd == "build-data" or (args.cmd == "build" and not args.skip_spark):
            build_data.run(catalog_size=args.catalog_size, top_tags=args.top_tags,
                           min_tag_tracks=args.min_tag_tracks, driver_memory=args.driver_memory)
        if args.cmd != "build-data":
            build_index.run()
        return
    if args.cmd == "train-als":
        from . import als
        als.run(app_feedback=args.feedback, rank=args.rank, max_iter=args.iters, reg=args.reg,
                alpha=args.alpha, holdout=args.holdout, seed=args.seed, driver_memory=args.driver_memory)
        return

    from .config import ARTIFACTS
    if not (ARTIFACTS / "index.npz").exists():
        sys.exit("No index found. Run:  uv run music-vibe extract && uv run music-vibe build")
    from .recommend import Recommender

    rec = Recommender()
    opts = dict(k=args.k, min_users=args.min_users, years=args.years, mood_mode=args.mood)
    if args.cmd == "ask":
        _print(*rec.recommend(" ".join(args.query), **opts), explain=args.explain)
        return
    print("Describe your mood / vibe (empty line to quit).")
    while True:
        try:
            text = input("\nvibe> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        _print(*rec.recommend(text, **opts), explain=args.explain)

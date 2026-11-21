"""Retrain implicit ALS on ListenBrainz feedback + every app user's thumbs up/down.

    uv run python web/manage.py train_als
    uv run python web/manage.py train_als --rank 64 --alpha 40

A running server picks up the new factors automatically on the next request.
"""

import pandas as pd
from django.core.management.base import BaseCommand

from recs.models import Feedback


class Command(BaseCommand):
    help = "Retrain the implicit ALS model (Spark) including web-app feedback."

    def add_arguments(self, parser):
        parser.add_argument("--rank", type=int, default=48)
        parser.add_argument("--iters", type=int, default=15)
        parser.add_argument("--reg", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=20.0)
        parser.add_argument("--holdout", type=float, default=0.1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--driver-memory", default="3g")

    def handle(self, *args, rank, iters, reg, alpha, holdout, seed, driver_memory, **opts):
        from music_vibe import als
        from music_vibe.config import ARTIFACTS

        rows = list(Feedback.objects.values("user_id", "recording_mbid", "value"))
        path = ARTIFACTS / "app_feedback.parquet"
        pd.DataFrame(rows, columns=["user_id", "recording_mbid", "value"]).to_parquet(path, index=False)
        self.stdout.write(f"Exported {len(rows)} feedback rows from {len({r['user_id'] for r in rows})} users")
        meta = als.run(app_feedback=path, rank=rank, max_iter=iters, reg=reg, alpha=alpha,
                       holdout=holdout, seed=seed, driver_memory=driver_memory)
        self.stdout.write(self.style.SUCCESS(
            f"ALS trained: hit-rate@50 {meta['hit_rate_at_50']} (popularity baseline "
            f"{meta['hit_rate_at_50_popularity']}) in {meta['seconds']}s"))

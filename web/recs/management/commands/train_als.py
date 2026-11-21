"""Retrain implicit ALS on all musicdb feedback after syncing the app's thumbs up/down into it
(musicdb table `feedback`, partition source=app).

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
        parser.add_argument("--plays", action="store_true", help="also train on Taste Profile play counts")

    def handle(self, *args, rank, iters, reg, alpha, holdout, seed, driver_memory, plays=False, **opts):
        from moodplay import als
        from moodplay import musicdb as db

        db.open_db()
        rows = list(Feedback.objects.values("user_id", "recording_mbid", "value", "updated"))
        df = pd.DataFrame(rows, columns=["user_id", "recording_mbid", "value", "updated"])
        df = df.assign(created=df.pop("updated").astype(str))
        if len(df):
            db.write("feedback", df, "app", "web app (Django recs.Feedback)")
        elif db.has("feedback", "app"):
            db.drop("feedback", "app")
        self.stdout.write(f"Synced {len(df)} feedback rows from {df.user_id.nunique()} users into musicdb")
        meta = als.run(rank=rank, max_iter=iters, reg=reg, alpha=alpha,
                       holdout=holdout, seed=seed, driver_memory=driver_memory, plays=plays)
        self.stdout.write(self.style.SUCCESS(
            f"ALS trained: hit-rate@50 {meta['hit_rate_at_50']} (popularity baseline "
            f"{meta['hit_rate_at_50_popularity']}) in {meta['seconds']}s"))

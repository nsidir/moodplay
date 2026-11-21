import os
import sys
import threading

from django.apps import AppConfig


class RecsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "recs"

    def ready(self):
        # Warm the recommender in the background so the first search is fast.
        # Under runserver only the reloaded child (RUN_MAIN=true) serves requests.
        serving = "runserver" in sys.argv and os.environ.get("RUN_MAIN") == "true"
        if serving or os.environ.get("MOVIE_VIBE_WARM") == "1":
            from .services import get_recommender

            threading.Thread(target=get_recommender, daemon=True).start()

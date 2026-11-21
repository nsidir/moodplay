"""Process-wide Recommender instance (the model and index load once, ~5s)."""

from __future__ import annotations

import os
import threading
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

_lock = threading.Lock()
_rec = None
_als_mtime = None


def get_recommender():
    global _rec, _als_mtime
    from moodplay.als import ALS_FILE
    from moodplay.recommend import Recommender

    with _lock:
        if _rec is None:
            _rec = Recommender()
            _als_mtime = ALS_FILE.stat().st_mtime if ALS_FILE.exists() else None
        else:
            # pick up a retrained ALS model without restarting the server
            mtime = ALS_FILE.stat().st_mtime if ALS_FILE.exists() else None
            if mtime != _als_mtime:
                _rec.reload_als()
                _als_mtime = mtime
    return _rec


def user_feedback(user) -> dict[str, int]:
    if not user.is_authenticated:
        return {}
    return dict(user.feedback.values_list("recording_mbid", "value"))

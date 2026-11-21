"""End-to-end tests against the real artifacts (run extract, build and train-als first).

    uv run python web/manage.py test recs
"""

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from .models import Feedback
from .services import get_recommender


class FlowTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user("alice", password="pw-123456!")

    def test_search_anonymous(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        r = self.client.get("/", {"q": "energetic workout, no metal"})
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.context["tracks"]), 5)
        self.assertIn("metal", r.context["info"]["excludes"])

    def test_artist_cap(self):
        tracks = self.client.get("/", {"q": "dreamy and atmospheric"}).context["tracks"]
        artists = [t.artist for t in tracks]
        self.assertTrue(all(artists.count(a) <= 2 for a in artists))

    def test_feedback_requires_login_and_validates(self):
        mbid = self.client.get("/", {"q": "chill"}).context["tracks"][0].mbid
        self.assertEqual(self.client.post("/feedback/", {"recording_mbid": mbid, "value": 1}).status_code, 401)
        self.client.force_login(self.user)
        self.assertEqual(self.client.post("/feedback/", {"recording_mbid": "nope", "value": 1}).status_code, 400)
        self.assertEqual(self.client.post("/feedback/", {"recording_mbid": mbid, "value": 3}).status_code, 400)

    def test_vote_toggle_and_history(self):
        self.client.force_login(self.user)
        mbid = self.client.get("/", {"q": "summer road trip"}).context["tracks"][0].mbid
        self.client.post("/feedback/", {"recording_mbid": mbid, "value": 1, "query": "summer road trip"})
        self.assertEqual(Feedback.objects.get(user=self.user, recording_mbid=mbid).value, 1)
        self.client.post("/feedback/", {"recording_mbid": mbid, "value": -1})
        self.assertEqual(Feedback.objects.get(user=self.user, recording_mbid=mbid).value, -1)
        self.client.post("/feedback/", {"recording_mbid": mbid, "value": 0})
        self.assertFalse(Feedback.objects.filter(user=self.user).exists())
        self.assertEqual(self.user.queries.count(), 1)

    def test_personalization(self):
        rec = get_recommender()
        if rec.Y is None:
            self.skipTest("ALS not trained")
        self.client.force_login(self.user)
        before = [t.mbid for t in self.client.get("/", {"q": "chill"}).context["tracks"]]
        # like a handful of well-known tracks that have ALS factors
        m = rec.meta
        has = rec.Y.any(axis=1)
        likes = m[has & m.artist.isin(["Radiohead", "Portishead", "Massive Attack"])].nlargest(6, "users")
        for mb in likes.recording_mbid:
            Feedback.objects.create(user=self.user, recording_mbid=mb, value=1)
        r = self.client.get("/", {"q": "chill"})
        after = r.context["tracks"]
        self.assertTrue(r.context["info"]["personalized"])
        self.assertNotEqual(before, [t.mbid for t in after])
        self.assertFalse(set(likes.recording_mbid) & {t.mbid for t in after}, "rated tracks must be hidden")

    def test_ratings_page_and_signup(self):
        self.client.force_login(self.user)
        mbid = get_recommender().meta.recording_mbid.iat[0]
        Feedback.objects.create(user=self.user, recording_mbid=mbid, value=1)
        self.assertContains(self.client.get("/my-ratings/"), get_recommender().meta.title.iat[0])
        self.client.logout()
        r = self.client.post("/signup/", {"username": "bob", "password1": "s3cret-pass-x!",
                                          "password2": "s3cret-pass-x!"})
        self.assertRedirects(r, "/")


def _tiny_db(tmp: Path):
    """A 5-track Last.fm musicdb: tracks 0-2 train, 3-4 test."""
    from moodplay import musicdb as db

    db.write("lastfm_tracks", pd.DataFrame({
        "tid": np.arange(5, dtype="int32"), "track_id": [f"TR{i}" for i in range(5)],
        "artist": ["a"] * 5, "title": [f"t{i}" for i in range(5)],
        "split": ["train"] * 3 + ["test"] * 2}))
    db.write("tags", pd.DataFrame({
        "item_id": ["TR0", "TR0", "TR1", "TR2", "TR3", "TR4"],
        "tag": ["Chill", "mellow", "chillout", "metal", "chill", "metal"],
        "count": [100, 50, 100, 100, 100, 100], "kind": "tag"}), "lastfm_ds")
    # test track 3 is close to the chill tracks 0/1 and to test track 4 (which has no train tags)
    db.write("similars", pd.DataFrame({"tid": [3, 3, 3, 4], "sim_tid": [0, 1, 4, 2],
                                       "score": [1.0, 0.8, 1.0, 1.0], "rank": [0, 1, 2, 0]}))


class MusicDbTests(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "musicdb"
        self._env = mock.patch.dict("os.environ", {"MUSICDB": str(self.root)})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_contract_enforced(self):
        from moodplay import musicdb as db

        with self.assertRaises(db.SchemaError):  # unknown column
            db.write("lastfm_tracks", pd.DataFrame({"tid": [0], "track_id": ["x"], "split": ["train"], "bogus": [1]}))
        with self.assertRaises(db.SchemaError):  # required column missing
            db.write("lastfm_tracks", pd.DataFrame({"tid": [0], "split": ["train"]}))
        with self.assertRaises(db.SchemaError):  # partitioned table needs a partition
            db.write("tags", pd.DataFrame({"item_id": ["x"], "tag": ["y"], "count": [1.0]}))
        self.assertFalse(db.has("lastfm_tracks"))  # failed writes leave nothing behind

    def test_roundtrip_partitions_nullable_fill_and_replace(self):
        from moodplay import musicdb as db

        _tiny_db(self.root)
        db.write("tags", pd.DataFrame({"item_id": ["m1"], "tag": ["sad"], "count": [3]}), "mb_recording")
        self.assertEqual(db.partitions("tags"), ["lastfm_ds", "mb_recording"])
        t = db.read("tags", sources=["mb_recording"])
        self.assertEqual(list(t.columns), db.CONTRACT["tags"].schema.names)  # contract order, incl. source
        self.assertTrue(t.kind.isna().all())                                   # missing nullable -> null
        self.assertEqual(t["count"].dtype, np.float32)
        # replacing one partition leaves the others alone
        db.write("tags", pd.DataFrame({"item_id": ["m2", "m3"], "tag": ["x", "y"], "count": [1, 2]}), "mb_recording")
        self.assertEqual(len(db.read("tags", sources=["mb_recording"])), 2)
        self.assertEqual(len(db.read("tags", sources=["lastfm_ds"])), 6)
        m = db.manifest()
        self.assertEqual(m["tables"]["tags"]["rows"], 8)
        self.assertEqual(m["schema_version"], db.SCHEMA_VERSION)
        self.assertIn("tags", m["contract"])

    def test_schema_evolution_new_column_reads_as_null(self):
        from moodplay import musicdb as db

        _tiny_db(self.root)
        cols = db.CONTRACT["lastfm_tracks"].cols
        with mock.patch.object(db.CONTRACT["lastfm_tracks"], "cols", cols + [db.Col("bpm", db.F32, "tempo")]):
            df = db.read("lastfm_tracks")
            self.assertIn("bpm", df.columns)
            self.assertTrue(df.bpm.isna().all())

    def test_check_detects_duplicates_and_pack_unpack(self):
        from moodplay import musicdb as db

        _tiny_db(self.root)
        problems = [p for p in db.check(verbose=False) if not p.endswith(": missing")]
        self.assertEqual(problems, [])
        db.write("similars", pd.DataFrame({"tid": [3, 3], "sim_tid": [0, 0], "score": [1.0, 1.0], "rank": [0, 1]}))
        self.assertTrue(any("duplicate" in p for p in db.check(verbose=False)))
        db.write("similars", pd.DataFrame({"tid": [3], "sim_tid": [0], "score": [1.0], "rank": [0]}))
        with mock.patch.object(db, "check", return_value=[]):
            out = db.pack(Path(self._tmp.name) / "x.tar")
        other = Path(self._tmp.name) / "restored" / "musicdb"
        other.parent.mkdir()
        with mock.patch.dict("os.environ", {"MUSICDB": str(other)}), mock.patch.object(db, "check", return_value=[]):
            db.unpack(out)
            self.assertEqual(len(db.read("lastfm_tracks")), 5)

    def test_newer_schema_refused(self):
        from moodplay import musicdb as db

        _tiny_db(self.root)
        m = db.manifest()
        m["schema_version"] = db.SCHEMA_VERSION + 1
        (self.root / "schema.json").write_text(__import__("json").dumps(m))
        with self.assertRaises(db.SchemaError):
            db.open_db()


class LastfmKnnTests(SimpleTestCase):
    """kNN tag propagation on a tiny synthetic musicdb (no real data needed)."""

    def test_train_uses_only_train_tags_and_evaluates_on_test(self):
        from moodplay import knn_tags as kt

        with tempfile.TemporaryDirectory() as tmp:
            root, art = Path(tmp) / "musicdb", Path(tmp) / "art"
            with mock.patch.dict("os.environ", {"MUSICDB": str(root)}), \
                    mock.patch.object(kt, "ARTIFACTS", art), \
                    mock.patch.object(kt, "KNN_TAGS", art / "knn_tags.parquet"), \
                    mock.patch.object(kt, "KNN_VOCAB", art / "knn_vocab.parquet"):
                _tiny_db(root)
                kt.train(min_tag_tracks=1)
                pred = pd.read_parquet(art / "knn_tags.parquet")
                p3 = pred[pred.tid == 3].sort_values("count", ascending=False)
                self.assertEqual(p3.tag.iat[0], "chill")             # chillout normalized, 2 neighbours agree
                self.assertNotIn("metal", set(p3.tag))               # test track 4's own tags are never used
                self.assertAlmostEqual(p3["count"].iat[0], 100 * 1.8 / (1.8 + kt.KNN_SHRINK), places=3)
                self.assertEqual(set(pred[pred.tid == 4].tag), {"metal"})
                res = kt.evaluate(ks=(1,))
            self.assertEqual(res["knn"]["precision@1"], 1.0)
            self.assertTrue((art / "eval_tags.json").exists())

    def test_auc(self):
        from moodplay.knn_tags import _auc

        self.assertEqual(_auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([True, True, False, False])), 1.0)
        self.assertEqual(_auc(np.array([0.5, 0.5]), np.array([True, False])), 0.5)
        self.assertIsNone(_auc(np.array([1.0]), np.array([True])))


class AudioTagsTests(SimpleTestCase):
    """Audio -> tag model on a synthetic musicdb: happy-sounding songs must get 'happy', sad ones 'sad'."""

    def test_learns_from_train_and_predicts_test(self):
        from moodplay import audio_tags as at
        from moodplay import musicdb as db

        rng = np.random.default_rng(0)
        n_tr, n_te = 300, 60
        n = n_tr + n_te
        happy = np.arange(n) % 2 == 0
        with tempfile.TemporaryDirectory() as tmp:
            root, art = Path(tmp) / "musicdb", Path(tmp) / "art"
            art.mkdir()
            patches = [mock.patch.dict("os.environ", {"MUSICDB": str(root)}),
                       *[mock.patch.object(at, k, art / v) for k, v in
                         (("AUDIO_TAGS", "a.parquet"), ("AUDIO_TAGS_LASTFM", "al.parquet"),
                          ("AUDIO_FEATURES", "af.parquet"), ("KNN_TAGS", "none.parquet"))],
                       mock.patch.object(at, "ARTIFACTS", art), mock.patch.object(at, "MIN_POS", 5)]
            for p in patches:
                p.start()
            try:
                keys = [f"artist {i}|song {i}" for i in range(n)]
                db.write("recordings", pd.DataFrame({"recording_mbid": [f"m{i}" for i in range(10)],
                                                     "title": [f"song {i}" for i in range(10)],
                                                     "song_key": keys[:10]}))
                db.write("lastfm_tracks", pd.DataFrame({
                    "tid": np.arange(n, dtype="int32"), "track_id": [f"TR{i}" for i in range(n)],
                    "split": ["train"] * n_tr + ["test"] * n_te, "song_key": keys}))
                db.write("tags", pd.DataFrame({"item_id": [f"TR{i}" for i in range(n)],
                                               "tag": np.where(happy, "happy", "sad"), "count": 100.0}), "lastfm_ds")
                hl = {db.ab_col(c, k): rng.random(n).astype("float32")
                      for c, ks in db.AB_CLASSIFIERS.items() for k in ks}
                hl[db.ab_col("mood_happy", "happy")] = np.where(happy, 0.9, 0.1) + rng.normal(0, 0.05, n)
                # MBIDs 0-9 are known recordings; the rest are linked only through the audio file's own tags
                db.write("ab_highlevel", pd.DataFrame({
                    "mbid": [f"m{i}" for i in range(n)], "submission_offset": 0,
                    "tag_artist": [f"Artist {i}" for i in range(n)], "tag_title": [f"Song {i}" for i in range(n)],
                    **hl}), "00")
                res = at.run(rebuild_features=True, ks=(1,))
                self.assertEqual(res["songs"]["test"], n_te)
                self.assertGreater(res["audio_all_test"]["precision@1"], 0.9)
                pred = pd.read_parquet(art / "a.parquet")  # only MBIDs in musicdb.recordings get catalogue tags
                self.assertEqual(set(pred.recording_mbid), {f"m{i}" for i in range(10)})
                top = pred.sort_values("count", ascending=False).groupby("recording_mbid").tag.first()
                self.assertEqual(top["m0"], "happy")
                self.assertEqual(top["m1"], "sad")
                self.assertTrue((art / "al.parquet").exists())
            finally:
                for p in patches:
                    p.stop()

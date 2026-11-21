"""End-to-end tests against the real artifacts (run extract, build and train-als first).

    uv run python web/manage.py test recs
"""

from django.contrib.auth.models import User
from django.test import TestCase

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

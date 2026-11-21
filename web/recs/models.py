from django.conf import settings
from django.db import models


class Feedback(models.Model):
    """A user's thumbs up (+1) / down (-1) on a recommended track (MusicBrainz recording MBID)."""

    UP, DOWN = 1, -1
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="feedback")
    recording_mbid = models.CharField(max_length=36)
    value = models.SmallIntegerField(choices=[(UP, "good"), (DOWN, "bad")])
    query = models.CharField(max_length=300, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "recording_mbid"], name="one_vote_per_track")]
        ordering = ["-updated"]

    def __str__(self):
        return f"{self.user} {'+' if self.value > 0 else '-'} {self.recording_mbid}"


class QueryLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="queries")
    text = models.CharField(max_length=300)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]

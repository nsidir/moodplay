from __future__ import annotations

import re

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from .models import Feedback, QueryLog
from .services import get_recommender, user_feedback

EXAMPLES = ["chill late night vibes", "I'm stressed", "energetic workout, no metal", "dreamy and atmospheric",
            "rainy sunday morning, acoustic", "focus music for coding", "summer road trip", "angry"]
MBID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def home(request):
    q = request.GET.get("q", "").strip()[:300]
    ctx = {"q": q, "examples": EXAMPLES, "mood": request.GET.get("mood", "lift"),
           "year_from": request.GET.get("year_from", ""), "year_to": request.GET.get("year_to", ""),
           "popular": request.GET.get("popular", "20")}
    if request.user.is_authenticated:
        ctx["recent"] = list(dict.fromkeys(request.user.queries.values_list("text", flat=True)[:30]))[:6]
    if q:
        rec = get_recommender()
        y0, y1 = _int(ctx["year_from"]), _int(ctx["year_to"])
        tracks, info = rec.recommend(
            q, k=15, min_users=_int(ctx["popular"], 20), years=(y0 or 0, y1 or 9999) if (y0 or y1) else None,
            mood_mode="match" if ctx["mood"] == "match" else "lift", feedback=user_feedback(request.user))
        if request.user.is_authenticated:
            QueryLog.objects.create(user=request.user, text=q)
        ctx.update(tracks=tracks, info=info, als=rec.als_meta)
    return render(request, "recs/home.html", ctx)


@require_POST
def feedback(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "login required"}, status=401)
    mbid, value = request.POST.get("recording_mbid", ""), _int(request.POST.get("value"))
    if not MBID.match(mbid) or value not in (-1, 0, 1):
        return HttpResponseBadRequest("bad recording_mbid/value")
    if mbid not in get_recommender().row_of:
        return HttpResponseBadRequest("unknown track")
    if value == 0:
        Feedback.objects.filter(user=request.user, recording_mbid=mbid).delete()
    else:
        Feedback.objects.update_or_create(user=request.user, recording_mbid=mbid,
                                          defaults={"value": value, "query": request.POST.get("query", "")[:300]})
    return JsonResponse({"ok": True, "value": value, "n_feedback": request.user.feedback.count()})


@login_required
def my_ratings(request):
    rec = get_recommender()
    rows = []
    for f in request.user.feedback.all():
        i = rec.row_of.get(f.recording_mbid)
        if i is None:
            continue
        m = rec.meta.iloc[i]
        rows.append({"fb": f, "title": m.title, "artist": m.artist,
                     "year": None if m.year != m.year else int(m.year)})
    ups = sum(1 for r in rows if r["fb"].value > 0)
    return render(request, "recs/ratings.html", {"rows": rows, "ups": ups, "downs": len(rows) - ups,
                                                 "als": rec.als_meta})


@login_required
@require_POST
def clear_history(request):
    request.user.queries.all().delete()
    messages.success(request, "Search history cleared.")
    return redirect("home")


def signup(request):
    form = UserCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.save())
        return redirect("home")
    return render(request, "registration/signup.html", {"form": form})

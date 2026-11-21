from django.contrib import admin

from .models import Feedback, QueryLog


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("user", "recording_mbid", "value", "query", "updated")
    list_filter = ("value",)
    search_fields = ("user__username", "query", "recording_mbid")


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ("user", "text", "created")
    search_fields = ("user__username", "text")

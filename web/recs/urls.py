from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("feedback/", views.feedback, name="feedback"),
    path("my-ratings/", views.my_ratings, name="my_ratings"),
    path("history/clear/", views.clear_history, name="clear_history"),
    path("signup/", views.signup, name="signup"),
]

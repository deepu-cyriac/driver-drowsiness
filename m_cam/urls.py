from django.urls import path
from . import views

urlpatterns = [
    #when someone calls for homepage
    path('',views.home, name='home'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('webcam_feed', views.webcam_feed, name='webcam_feed' )
]
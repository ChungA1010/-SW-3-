from django.urls import path
from . import views

app_name = 'preprocessing'

urlpatterns = [
    path('audio/', views.upload_audio, name='audio'),
    path('video/',views.upload_video, name='video'),
]
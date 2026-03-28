from django.urls import path
from . import views

app_name = 'source_separator'

urlpatterns = [
    path('upload/', views.upload_audio, name='upload'),
]
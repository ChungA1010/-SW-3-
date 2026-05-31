import os
from django.apps import AppConfig
from django.conf import settings

class AiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai'
    
    def ready(self):
        pass
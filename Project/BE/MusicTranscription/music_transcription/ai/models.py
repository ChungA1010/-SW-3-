from django.db import models
from preprocessing.models import SeparatedTrack

# Create your models here.
class RecordedTrack(models.Model):
    source = models.ForeignKey(SeparatedTrack, 
                               on_delete=models.CASCADE,
                               related_name='recorded_tracks')
    file = models.FileField(upload_to='uploads/recorded_audio/%Y/%m/%d/')
    created_at = models.DateTimeField(auto_now_add=True)


class Effector(models.Model):
    class Level(models.TextChoices):
        OFF = 'off', 'Off'
        HALF = '50', '50%'
        FULL = '100', '100%'
    
    source = models.ForeignKey(SeparatedTrack, 
                               on_delete=models.CASCADE,
                               related_name='effector_values')
    distortion = models.CharField(
        max_length=3,
        choices=Level.choices,
        null=True,
        default=None,
        blank=True
    )
    delay = models.CharField(
        max_length=3,
        choices=Level.choices,
        null=True,
        default=None,
        blank=True
    )
    phase = models.CharField(
        max_length=3,
        choices=Level.choices,
        null=True,
        default=None,
        blank=True
    )
    
    
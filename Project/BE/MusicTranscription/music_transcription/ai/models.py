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
    source = models.ForeignKey(SeparatedTrack, 
                               on_delete=models.CASCADE,
                               related_name='effector_values')
    
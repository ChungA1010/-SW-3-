from django.db import models
import uuid
from django.utils.timezone import now
from django.utils import timezone

def track_directory_path(instance, filename):
    #based on when the source file is uploaded
    date_path = instance.source.created_at.strftime('%Y/%m/%d')
    # ex: uploads/separated_audio/2026/03/28/12312/vocals.wav
    return f'uploads/separated_audio/{date_path}/{instance.source.id}/{filename}'

class SourceAudio(models.Model):
    """model storing information about source audio file uploaded by user"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, help_text="name of the uploaded audio file")
    file = models.FileField(upload_to='uploads/source_audio/%Y/%m/%d/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    #possible asynchronous states 
    #not necessary right now, but maybe later for asynchronous processing
    status = models.CharField(
        max_length=20, 
        choices=[('PENDING', '대기중'), ('PROCESSING', '변환중'),
                 ('COMPLETED', '완료'), ('FAILED', '실패')],
        default='PENDING'
    )
    
    def __str__(self):
        return self.name

class SeparatedTrack(models.Model):
    source = models.ForeignKey(SourceAudio, 
                               on_delete=models.CASCADE,
                               related_name='separated_tracks')
    
    TRACK_CHOICES = (
        ('vocals', 'Vocals'),
        ('drums', 'Drums'),
        ('bass', 'Bass'),
        ('other', 'Other'),
    )
    
    track_type = models.CharField(max_length=10, choices=TRACK_CHOICES)
    file = models.FileField(upload_to=track_directory_path)
    
    
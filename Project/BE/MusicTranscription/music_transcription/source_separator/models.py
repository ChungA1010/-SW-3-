from django.db import models
import uuid


class SourceAudio(models.Model):
    """model storing information about source audio file uploaded by user"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, help_text="name of the uploaded audio file")
    file = models.FileField(upload_to='uploads/source_audio/%Y/%m/%d/')
    
    #possible asynchronous states
    status = models.CharField(
        max_length=20, 
        choices=[('PENDING', '대기중'), ('PROCESSING', '변환중'), ('COMPLETED', '완료'), ('FAILED', '실패')],
        default='PENDING'
    )
    
    def __str__(self):
        return self.title
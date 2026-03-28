from django import forms
from .models import SourceAudio

class SourceAudioForm(forms.ModelForm):
    class Meta:
        model = SourceAudio
        # 사용자에게 입력받을 필드만 지정합니다. (status는 자동으로 PENDING이 됩니다)
        fields = ['name', 'file']
from django import forms
from .models import SourceAudio

class SourceAudioForm(forms.ModelForm):
    start_sec = forms.FloatField(required=False, label="시작 시간(초)")
    end_sec = forms.FloatField(required=False, label="종료 시간(초)")
    
    class Meta:
        model = SourceAudio
        # 사용자에게 입력받을 필드만 지정합니다. (status는 자동으로 PENDING이 됩니다)
        fields = ['name', 'file',]
    
    def clean(self):
        cleaned_data = super().clean()
        start_sec = cleaned_data.get('start_sec')
        end_sec = cleaned_data.get('end_sec')

        # 둘 중 하나만 입력된 경우나, 시작이 종료보다 늦은 경우 에러 처리
        if start_sec is not None and end_sec is not None:
            if start_sec >= end_sec:
                raise forms.ValidationError("종료 시간은 시작 시간보다 커야 합니다.")
        elif (start_sec is not None and end_sec is None) or (start_sec is None and end_sec is not None):
            raise forms.ValidationError("시작 시간과 종료 시간을 모두 입력하거나, 모두 비워두세요.")

        return cleaned_data

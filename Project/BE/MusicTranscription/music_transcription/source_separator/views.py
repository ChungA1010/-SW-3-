from django.shortcuts import render
from django.http import JsonResponse
from .forms import SourceAudioForm

from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def upload_audio(request):
    if request.method == 'POST':
        form = SourceAudioForm(request.POST, request.FILES)
        
        if form.is_valid():
            source_audio = form.save()
            return JsonResponse({"status": "success", "message": "Upload complete", "id": str(source_audio.id)})
    else:
        form = SourceAudioForm()

    



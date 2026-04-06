import os
import sys
import shutil
import subprocess
from django.conf import settings
from django.http import JsonResponse
from django.core.files import File
from django.views.decorators.csrf import csrf_exempt

from .models import SeparatedTrack
from .forms import SourceAudioForm

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST':
        form = SourceAudioForm(request.POST, request.FILES)
        
        if form.is_valid():
            #save original source file
            source_audio = form.save()
            
            #update state
            source_audio.status = 'PROCESSING'
            source_audio.save()
            
            #get path
            input_path = source_audio.file.path
            
            #create temporary directory to save separated files
            #(going to save again using properly as File object)
            temp_output_dir = os.path.join(settings.MEDIA_ROOT, 'demucs_temp')
            os.makedirs(temp_output_dir, exist_ok=True)
            
            #Demucs command
            command = [
                sys.executable, '-m', 'demucs.separate',
                '-n', 'htdemucs',
                '--shifts', '2', 
                '-o', temp_output_dir,
                input_path
            ]
            
            try:
                #run Demucs command
                subprocess.run(command, check=True)
                
                #get temporary directory path
                filename_no_ext = os.path.splitext(os.path.basename(input_path))[0]
                result_dir = os.path.join(temp_output_dir, 'htdemucs', filename_no_ext)
                
                track_types = ['vocals', 'drums', 'bass', 'other']
                saved_track_urls = {}
                
                #save separated files to SeparatedTrack model
                for t_type in track_types:
                    file_path = os.path.join(result_dir, f'{t_type}.wav')
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            track = SeparatedTrack(source=source_audio, track_type=t_type)
                            track.file.save(f'{source_audio.id}_{t_type}.wav', File(f))
                            #save url for frontend
                            saved_track_urls[t_type] = track.file.url
                
                # remove temporary directory
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
                
                #update state to COMPLETED
                source_audio.status = 'COMPLETED'
                source_audio.save()
                
                #return success response
                return JsonResponse({
                    "status": "success", 
                    "message": "음원 분리가 완료되었습니다.", 
                    "source_id": str(source_audio.id), 
                    "tracks": saved_track_urls
                })
                
            except subprocess.CalledProcessError as e:
                # update state to FAILED 
                source_audio.status = 'FAILED'
                source_audio.save()
                
                return JsonResponse({
                    "status": "error", 
                    "message": f"Demucs 처리에 실패했습니다: {str(e)}"
                }, status=500)
        
        else:
            #if form is not valid
            return JsonResponse({"status": "error", "errors": form.errors}, status=400)
                
    else:
        return JsonResponse({"status": "error", "message": "잘못된 요청 방식입니다."}, status=405)

    



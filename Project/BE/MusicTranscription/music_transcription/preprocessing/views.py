import os
import sys
import shutil
import subprocess
from django.conf import settings
from django.http import JsonResponse
from django.core.files import File
from django.views.decorators.csrf import csrf_exempt

from pydub import AudioSegment

from .models import SeparatedTrack
from .forms import SourceAudioForm

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST':
        form = SourceAudioForm(request.POST, request.FILES)
        
        if form.is_valid():
            #save original source file
            source_audio = form.save()
            input_path = source_audio.file.path
            
            start_sec = form.cleaned_data.get('start_sec')
            end_sec = form.cleaned_data.get('end_sec')
            
            #create temporary directory to save separated files
            #(going to save again using properly as File object)
            temp_output_dir = os.path.join(settings.MEDIA_ROOT, 'demucs_temp')
            os.makedirs(temp_output_dir, exist_ok=True)
            
            #cut audio first
            demucs_input_path = input_path
            temp_cut_file = None
            
            if start_sec is not None and end_sec is not None:
                audio = AudioSegment.from_file(input_path)
                start_ms = int(start_sec * 1000)
                end_ms = int(end_sec * 1000)
                cut_audio = audio[start_ms:end_ms]
                
                temp_cut_file = os.path.join(temp_output_dir, f'cut_temp_{source_audio.id}.wav')
                cut_audio.export(temp_cut_file, format="wav")
                
                demucs_input_path = temp_cut_file
            
            #Demucs command
            command = [
                sys.executable, '-m', 'demucs.separate',
                '-n', 'htdemucs',
                '--shifts', '2', 
                '-o', temp_output_dir,
                demucs_input_path
            ]
            
            try:
                #run Demucs command
                subprocess.run(command, check=True)
                
                #get temporary directory path
                filename_no_ext = os.path.splitext(os.path.basename(demucs_input_path))[0]
                result_dir = os.path.join(temp_output_dir, 'htdemucs', filename_no_ext)
                
                
                #save separated files to SeparatedTrack model
                target_track_type = 'other'
                file_path = os.path.join(result_dir, f'{target_track_type}.wav')
                saved_track_url = None
                
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        track = SeparatedTrack(source=source_audio, track_type=target_track_type)
                        track.file.save(f'{source_audio.id}_{target_track_type}.wav', File(f))
                        saved_track_url = track.file.url
                
                # remove temporary directory
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
                if temp_cut_file and os.path.exists(temp_cut_file):
                    os.remove(temp_cut_file)
                
                source_audio.save()
                
                # 'other' 트랙이 성공적으로 추출되었는지 확인 후 응답
                if saved_track_url:
                    return JsonResponse({
                        "status": "success", 
                        "message": "악기 분리 및 저장이 완료되었습니다.", 
                        "source_id": str(source_audio.id), 
                        "guitar_url": saved_track_url  
                    })
                else:
                    return JsonResponse({
                        "status": "error", 
                        "message": "분리된 파일에서 'other' 트랙을 찾을 수 없습니다."
                    }, status=500)
                
            except subprocess.CalledProcessError as e:
                if temp_cut_file and os.path.exists(temp_cut_file):
                    os.remove(temp_cut_file)
                
                return JsonResponse({
                    "status": "error", 
                    "message": f"Demucs 처리에 실패했습니다: {str(e)}"
                }, status=500)
        
        else:
            #if form is not valid
            return JsonResponse({"status": "error", "errors": form.errors}, status=400)
                
    else:
        return JsonResponse({"status": "error", "message": "잘못된 요청 방식입니다."}, status=405)

    



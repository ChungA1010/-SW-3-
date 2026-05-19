import os
import sys
import shutil
import subprocess
import json
from django.conf import settings
from django.http import JsonResponse
from django.core.files import File
from django.views.decorators.csrf import csrf_exempt

from pydub import AudioSegment
import yt_dlp

from .models import SeparatedTrack, SourceAudio
from .forms import SourceAudioForm, SourceVideoForm

@csrf_exempt
def upload_audio(request):
    """
    1. 파일을 저장하고 자릅니다.
    2. Demucs를 돌려 'other' 트랙을 추출합니다.
    3. 완료된 파일의 경로와 URL 정보를 딕셔너리로 반환합니다.
    """
    form = SourceAudioForm(request.POST, request.FILES)
    if not form.is_valid():
        return {"success": False, "error": form.errors, "status": 400}

    source_audio = form.save()
    input_path = source_audio.file.path
    start_sec = form.cleaned_data.get('start_sec')
    end_sec = form.cleaned_data.get('end_sec')
    
    temp_output_dir = os.path.join(settings.MEDIA_ROOT, 'demucs_temp')
    os.makedirs(temp_output_dir, exist_ok=True)
    
    demucs_input_path = input_path
    temp_cut_file = None
    
    # 오디오 자르기
    if start_sec is not None and end_sec is not None:
        audio = AudioSegment.from_file(input_path)
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        cut_audio = audio[start_ms:end_ms]
        
        temp_cut_file = os.path.join(temp_output_dir, f'cut_temp_{source_audio.id}.wav')
        cut_audio.export(temp_cut_file, format="wav")
        demucs_input_path = temp_cut_file

    # Demucs 명령어 실행
    command = [
        sys.executable, '-m', 'demucs.separate',
        '-n', 'htdemucs',
        '-o', temp_output_dir,
        demucs_input_path
    ]
    
    try:
        subprocess.run(command, check=True)
        
        filename_no_ext = os.path.splitext(os.path.basename(demucs_input_path))[0]
        result_dir = os.path.join(temp_output_dir, 'htdemucs', filename_no_ext)
        
        target_track_type = 'other'
        file_path = os.path.join(result_dir, f'{target_track_type}.wav')
        
        if not os.path.exists(file_path):
            raise Exception("분리된 파일에서 'other' 트랙을 찾을 수 없습니다.")

        # SeparatedTrack에 저장
        with open(file_path, 'rb') as f:
            track = SeparatedTrack(source=source_audio, track_type=target_track_type)
            track.file.save(f'{source_audio.id}_{target_track_type}.wav', File(f))
            
        # [성공 시 반환할 데이터]
        result_data = {
            "success": True,
            "track_path": track.file.path, # AI 모델에 넣을 경로
            "stem_url": track.file.url,    # 프론트로 보낼 경로
            "original_name": os.path.basename(source_audio.file.name),
            "original_url": source_audio.file.url
        }

    except Exception as e:
        result_data = {"success": False, "error": f"Demucs 실패: {str(e)}", "status": 500}
        
    finally:
        # 성공하든 에러가 나든 임시 파일은 무조건 삭제 (try-finally 활용)
        if 'result_dir' in locals() and os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        if temp_cut_file and os.path.exists(temp_cut_file):
            os.remove(temp_cut_file)
            
    return result_data


@csrf_exempt
def upload_video(request):
    """
    유튜브 링크를 받아 mp3로 다운받고, Demucs를 돌린 후 결과 경로를 반환합니다.
    """
    # 1. JSON 데이터 파싱 (프론트에서 JSON으로 보내기 때문)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = request.POST # 예외: 만약 form-data로 왔다면 이걸 씀

    form = SourceVideoForm(data)
    
    if not form.is_valid():
        return {"success": False, "error": form.errors, "status": 400}

    url = form.cleaned_data['url']
    start_sec = form.cleaned_data.get('start_sec')
    end_sec = form.cleaned_data.get('end_sec')
    
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'yt_temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    downloaded_file = None
    temp_cut_file = None
    result_dir = None
    
    try:
        # [Step 1] 유튜브 음원 추출 (사용자 원본 코드 유지)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            base_filename = ydl.prepare_filename(info_dict)
            downloaded_file = os.path.splitext(base_filename)[0] + '.wav'
        
        # SourceAudio에 저장
        source_audio = SourceAudio()
        with open(downloaded_file, 'rb') as f:
            file_name = f"yt_{info_dict['id']}.wav"
            source_audio.file.save(file_name, File(f))
        source_audio.save()
        
        input_path = source_audio.file.path
        demucs_input_path = input_path
        
        # [Step 2] 자르기 로직
        if start_sec is not None and end_sec is not None:
            audio = AudioSegment.from_file(input_path)
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            cut_audio = audio[start_ms:end_ms]
            
            temp_cut_file = os.path.join(temp_dir, f'cut_temp_{source_audio.id}.wav')
            cut_audio.export(temp_cut_file, format="wav")
            demucs_input_path = temp_cut_file
            
        # [Step 3] Demucs 실행
        command = [
            sys.executable, '-m', 'demucs.separate',
            '-n', 'htdemucs',
            '--shifts','2',
            '-o', temp_dir,
            demucs_input_path
        ]
        subprocess.run(command, check=True)
        
        filename_no_ext = os.path.splitext(os.path.basename(demucs_input_path))[0]
        result_dir = os.path.join(temp_dir, 'htdemucs', filename_no_ext)
        
        target_track_type = 'other'
        file_path = os.path.join(result_dir, f'{target_track_type}.wav')
        
        if not os.path.exists(file_path):
            raise Exception("분리된 파일에서 'other' 트랙을 찾을 수 없습니다.")

        # SeparatedTrack 저장
        with open(file_path, 'rb') as f:
            track = SeparatedTrack(source=source_audio, track_type=target_track_type)
            track.file.save(f'{source_audio.id}_{target_track_type}.wav', File(f))
            
        # ✅ 성공 시 반환 딕셔너리 (파일 업로드 결과와 똑같은 규격!)
        return {
            "success": True,
            "track_path": track.file.path,
            "stem_url": track.file.url,
            "original_name": file_name,
            "original_url": source_audio.file.url
        }
            
    except Exception as e:
        return {"success": False, "error": f"YouTube 처리 중 오류: {str(e)}", "status": 500}
        
    finally:
        # 무조건 실행되는 쓰레기 파일 정리 로직
        if result_dir and os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        if temp_cut_file and os.path.exists(temp_cut_file):
            os.remove(temp_cut_file)
        if downloaded_file and os.path.exists(downloaded_file):
            os.remove(downloaded_file)
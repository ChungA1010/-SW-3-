import requests
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os

from preprocessing.views import upload_audio, upload_video

# 전처리 함수 및 모델 클래스 임포트
from preprocessing.models import SeparatedTrack
from .models import Effector, RecordedTrack


COLAB_API_URL = "https://blissful-entrench-donut.ngrok-free.dev/predict"


@csrf_exempt
def run_ai_inference(audio_path):
    """
    로컬 대신 구글 코랩 GPU 서버로 오디오를 전송하여 결과를 받아옵니다.
    """
    try:
        # 1. 파일 열기
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            
            # 2. 코랩 서버로 POST 요청 보내기
            response = requests.post(COLAB_API_URL, files=files, timeout=300)
            
        # 3. 결과 파싱
        if response.status_code == 200:
            api_result = response.json()
            if api_result.get("success"):
                return {
                    "success": True,
                    "result": api_result.get("result"),

                }
            else:
                return {"success": False, "error": f"코랩 내부 에러: {api_result.get('error')}"}
        else:
            return {"success": False, "error": f"코랩 연결 실패 (Status: {response.status_code})"}
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "AI 서버 응답 시간 초과 (Timeout)"}
    except Exception as e:
        return {"success": False, "error": f"네트워크 통신 실패: {str(e)}"}
    
    
@csrf_exempt
def analyze_ai(request):
    if request.method != 'POST':
        return JsonResponse({"success": False, "message": "POST 요청만 허용됩니다."}, status=405)

    # 1. Content-Type을 보고 유튜브인지 파일인지 판단하여 전처리 실행
    if request.content_type == 'application/json':
        prep_result = upload_video(request)
    else:
        prep_result = upload_audio(request)
    
    # 2. 전처리(음원 분리 등) 실패 시 에러 응답
    if not prep_result["success"]:
        return JsonResponse({
            "success": False, 
            "message": prep_result["error"]
        }, status=prep_result.get("status", 400))

    # 3. 전처리가 완료된 오디오 경로(track_path)를 코랩 AI 전송함수로 전달
    ai_result = run_ai_inference(prep_result["track_path"])
    
    #이펙터 값 데이터베이스에 저장
    result_data = json.loads(ai_result["result"]) # json response를 dict로 변환
    
    dist = result_data.get("dist")
    delay = result_data.get("delay")
    phase = result_data.get("phase")
    
    effector = Effector(
        dist=dist,
        delay=delay,
        phase=phase
    )
    
    effector.save()
    
    
    # 4. AI 추론 실패 시 에러 응답
    if not ai_result["success"]:
        return JsonResponse({
            "success": False, 
            "message": ai_result.get("error", "알 수 없는 에러 발생")
        }, status=500)

    return JsonResponse({
        "success": True,
        "source_name":prep_result["original_name"],
        "predicted_effect": ai_result.get("result"),
        "guitar_stem_url": prep_result["stem_url"],
        "original_audio_url": prep_result["original_url"],
    })
    
@csrf_exempt
def return_feedback(request):
    if request.method != 'POST':
        return JsonResponse({"success": False, "message": "POST 요청만 허용됩니다."}, status=405)
    if 'file' not in request.FILES:
        return JsonResponse({"success": False, "message": "파일('file' key)이 누락되었습니다."}, status=400)
    
    uploaded_file = request.FILES['file']
    
    
    try:
        # (주의) 아까 만든 모델에 외래키(source)가 필수이므로, 
        # 실제 구현 시에는 프론트에서 부모 ID를 받아와서 조회를 먼저 해야 합니다.
        # 여기서는 예시로 가장 최근의 SeparatedTrack을 매핑했다고 가정하겠습니다.
        dummy_source = SeparatedTrack.objects.latest('id') 

        # 📌 3. 데이터베이스에 저장하기
        recorded_track = RecordedTrack(
            source=dummy_source,
            file=uploaded_file # 👈 장고 FileField는 UploadedFile 객체를 주면 알아서 파일로 저장합니다.
        )
        recorded_track.save()

        # 📌 4. 프론트엔드가 기다리는 SimpleResponse 형태로 응답
        return JsonResponse({
            "success": True, 
            "message": f"'{dummy_source.id}' 파일이 백엔드에 무사히 저장되었습니다!"
        })

    except Exception as e:
        return JsonResponse({
            "success": False, 
            "message": f"서버 처리 중 오류 발생: {str(e)}"
        }, status=500)    
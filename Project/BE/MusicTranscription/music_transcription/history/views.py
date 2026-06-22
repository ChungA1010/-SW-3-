from django.http import JsonResponse
from preprocessing.models import SeparatedTrack

def get_all_logs(request):
    if request.method == 'GET':
        # - source(SourceAudio)는 정방향 참조이므로 select_related 사용
        # - effector_values(Effector)는 역방향 참조이므로 prefetch_related 사용
        tracks = SeparatedTrack.objects.select_related('source') \
                                       .prefetch_related('effector_values') \
                                       .all().order_by('-id')  # 최신순 
        logs = []
        for track in tracks:
            # 해당 트랙에 연결된 Effector 값 가져오기 (.first() 사용)
            effector = track.effector_values.first()
            
            # 데이터를 계층형 딕셔너리로 묶기
            logs.append({
                "track_id": track.id,
                # 부모 테이블(SourceAudio)의 데이터
                "source_info": {
                    "source_id": str(track.source.id), 
                    "name": track.source.name,
                },
                # 분리된 음원 파일 경로
                "file_path": track.file.url if track.file else None,

                # 자식 테이블(Effector)의 데이터
                "predicted_effect": {
                    "dist": effector.dist if effector else "off",
                    "delay": effector.delay if effector else "off",
                    "phase": effector.phase if effector else "off",
                } if effector else None
            })
            
        return JsonResponse({"success": True, "logs": logs})
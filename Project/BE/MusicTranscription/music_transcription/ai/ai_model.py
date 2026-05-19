# ai/ai_model.py 수정

import torch
import librosa
from .apps import AiConfig  # 메모리에 올라간 모델 가져오기

def predict_effect(audio_path):
    """
    새로운 Hugging Face 모델을 이용해 이펙터를 예측합니다.
    """
    try:
        # 1. 오디오 로드 (HF 모델은 보통 16000Hz를 표준으로 사용합니다)
        # 📌 기존 audio_utils.py의 복잡한 로직 대신 librosa로 바로 읽습니다.
        speech, sr = librosa.load(audio_path, sr=16000)

        # 2. 모델과 프로세서 꺼내기
        processor = AiConfig.processor
        model = AiConfig.model

        # 3. 전처리 (Processor가 알아서 텐서로 변환해 줌)
        inputs = processor(speech, sampling_rate=sr, return_tensors="pt")

        # 4. 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 5. 결과 해석
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
        # 모델의 config.json에 저장된 라벨 이름 꺼내기
        predicted_label = model.config.id2label[predicted_class_id]

        # 확률(Confidence) 계산
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class_id].item()

        return {
            "success": True,
            "predicted_effect_display_name": predicted_label,
            "confidence": round(confidence * 100, 2) # 퍼센트로 보기 좋게 변환
        }

    except Exception as e:
        print(f"예측 중 에러 발생: {e}")
        return {"success": False, "error": str(e)}
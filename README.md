# Effect Analyzer Complete

Next.js 프론트엔드와 FastAPI 백엔드, Demucs 기타 분리, 기존 GRU 모델 추론을 하나로 묶은 통합 프로젝트입니다.

## 1. 아키텍처 요약

- `frontend`: Next.js + React + TypeScript UI
- `backend`: FastAPI API 서버
- `backend/app/services/youtube_service.py`: `yt-dlp` 기반 유튜브 오디오 수집
- `backend/app/services/demucs_service.py`: Demucs 실행 및 기타 stem 선택
- `backend/app/services/inference_service.py`: 기존 `09_gru` 체크포인트를 로드해 추론
- `backend/app/services/audio_preprocess.py`: GRU 입력 전처리

분석 흐름:

1. 사용자가 파일을 업로드하거나 유튜브 링크를 입력합니다.
2. 백엔드가 원본 오디오를 `uploads/` 또는 `downloads/` 에 저장합니다.
3. Demucs가 오디오를 분리합니다.
4. 설정된 stem 후보(`guitar`, `other`) 중 실제 존재하는 stem을 선택합니다.
5. 선택된 stem을 기존 GRU 전처리 규격으로 MFCC + delta + delta2 시퀀스로 변환합니다.
6. GRU 모델이 fine effect를 추론하고 coarse family 점수까지 합산합니다.
7. 프론트엔드가 예측 라벨, 점수, 오디오 플레이어를 표시합니다.

## 2. 폴더 구조

```text
effect-analyzer-complete/
  frontend/
    app/
      globals.css
      layout.tsx
      page.tsx
    components/
      analyzer-page.tsx
    lib/
      api.ts
      types.ts
    .env.local.example
    next-env.d.ts
    next.config.mjs
    package.json
    tsconfig.json
  backend/
    app/
      api/
        analyze.py
      core/
        config.py
        errors.py
      models/
        gru_model.py
      schemas/
        analyze.py
      services/
        audio_preprocess.py
        demucs_service.py
        inference_service.py
        pipeline_service.py
        youtube_service.py
      utils/
        file_utils.py
        validators.py
      main.py
    weights/
      gru_model.pt
    uploads/
    downloads/
    separated/
    temp/
    .env.example
    requirements.txt
  .gitignore
  README.md
```

## 3. 설치 방법

### 3-1. Python 백엔드

```bash
cd backend
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

의존성 설치:

```bash
pip install -r requirements.txt
```

`.env.example` 을 `.env` 로 복사하고 필요 시 수정합니다.

### 3-2. Node 프론트엔드

```bash
cd frontend
npm install
```

`.env.local.example` 을 `.env.local` 로 복사합니다.

## 4. 실행 방법

### 4-1. 백엔드 실행

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4-2. 프론트엔드 실행

```bash
cd frontend
npm run dev
```

브라우저에서 `http://localhost:3000` 접속

헬스 체크:

```bash
curl http://localhost:8000/api/health
```

## 5. API 요약

### `POST /api/analyze/file`

- multipart form-data
- 필드: `file`

예상 응답:

```json
{
  "success": true,
  "source_type": "file",
  "source_name": "my_song.wav",
  "predicted_effect": "Chorus",
  "predicted_effect_display_name": "Chorus",
  "predicted_family": "phase",
  "predicted_family_display_name": "Phase",
  "fine_scores": [],
  "family_scores": [],
  "guitar_stem_url": "/media/separated/xxxx_other.wav",
  "original_audio_url": "/media/uploads/xxxx/input.wav",
  "processing_time_sec": 8.42,
  "demucs_stem_used": "other",
  "model_name": "gru"
}
```

### `POST /api/analyze/youtube`

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=..."
}
```

### `GET /api/health`

- 모델 weight 존재 여부와 Demucs 모델 이름을 반환합니다.

## 6. Demucs / yt-dlp / ffmpeg 주의사항

### Demucs

- 이 프로젝트는 `python -m demucs.separate` 명령으로 Demucs를 실행합니다.
- CPU 기본값으로 설정되어 있습니다.
- GPU 사용 시 `backend/.env` 에서 `DEMUCS_DEVICE=cuda` 로 변경하세요.

### stem 선택

- 대부분의 Demucs 모델은 `vocals / drums / bass / other` 를 출력합니다.
- 기타 전용 stem이 없는 경우 이 프로젝트는 기본적으로 `other.wav` 를 기타 입력으로 사용합니다.
- 만약 본인 환경의 Demucs 모델이 `guitar.wav` 를 출력한다면 `DEMUCS_STEM_CANDIDATES=guitar,other` 순서 그대로 사용하면 됩니다.
- 더 정확한 기타 추출이 필요하면 이 값과 Demucs 모델 종류를 직접 맞추세요.

### yt-dlp

- `yt-dlp` 는 Python 패키지로 설치됩니다.
- 실제 오디오 변환에는 `ffmpeg` 가 필요합니다.

ffmpeg 설치가 안 되어 있으면:

- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`

## 7. 수정해야 하는 포인트

### 모델 경로

- 파일: `backend/app/core/config.py`
- 항목: `resolved_weights_path`
- 필요 시 `.env` 의 `WEIGHTS_PATH` 로 변경

### 라벨 매핑

- 파일: `backend/app/core/config.py`
- 항목:
  - `fine_class_names`
  - `coarse_class_names`
  - `fine_to_coarse`
  - `label_display_names`

### 전처리 방식

- 파일: `backend/app/services/audio_preprocess.py`
- 현재는 기존 `09_gru` 규격과 동일하게:
  - sample rate: 32000
  - segment: 3.0초
  - feature: MFCC + delta + delta2
  - hop length: 256

### Demucs stem 선택

- 파일: `backend/app/services/demucs_service.py`
- 현재는 `.env` 의 `DEMUCS_STEM_CANDIDATES` 순서대로 stem을 찾습니다.
- 주석으로 수정 지점을 남겨두었습니다.

### 유튜브 검증 규칙

- 파일: `backend/app/utils/validators.py`
- `YOUTUBE_PATTERN` 수정 가능

## 8. 임시 파일 정리 정책

- 업로드, 다운로드, 분리 결과, temp 폴더는 요청이 들어올 때마다 TTL 기준으로 오래된 파일을 자동 정리합니다.
- 설정 위치: `backend/.env` 의 `TEMP_FILE_TTL_HOURS`

## 9. 저작권 및 플랫폼 정책 주의

- 유튜브 링크 입력 기능은 사용자가 적법하게 접근 가능한 콘텐츠만 분석한다는 전제입니다.
- 실제 서비스 운영 시 저작권, 플랫폼 정책, 사내 보안 정책을 별도로 검토하세요.

## 10. 권장 점검 항목

1. `backend/weights/gru_model.pt` 가 현재 사용하는 최신 GRU 체크포인트인지 확인
2. Demucs가 본인 환경에서 어떤 stem 이름을 출력하는지 확인
3. `ffmpeg` 가 설치되어 있는지 확인
4. 프론트 `.env.local` 의 API 주소가 실제 백엔드 주소와 일치하는지 확인
5. 샘플 wav/mp3 와 실제 유튜브 링크로 각각 한 번씩 테스트


# 06ver

`05ver`를 안전하게 보존한 채, `02ver/backend`와 실제로 연결하는 실험용 프론트입니다.

## 실행 순서

1. `02ver/backend` 실행

```bash
cd 02ver/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

2. `06ver` 환경 변수 설정

```bash
cp .env.local.example .env.local
```

기본값은 `http://localhost:8000/api` 입니다.

3. `06ver` 프론트 실행

```bash
cd 06ver
npm install
npm run dev
```

## 현재 연결 범위

- `분석 요청 창`의 `파일 업로드`, `링크 입력`이 실제 `POST /api/jobs`로 연결됩니다.
- 프론트는 `GET /api/jobs/{job_id}`를 폴링하고, 완료되면 `GET /api/results/{job_id}`를 가져옵니다.
- 결과는 `악기별 탐색`, `악보`, `이펙터`, `비교분석` 페이지에 반영됩니다.
- `샘플 기반` 모드는 기존 디자인/인터랙션 흐름을 유지합니다.

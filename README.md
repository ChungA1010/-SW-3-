# OSS-Project

공개소프트웨어 프로젝트 저장소입니다.  
본 프로젝트는 Git/GitHub 기반 협업 방식을 학습하고, 팀원들과 함께 기능을 개발 및 관리하는 것을 목표로 합니다.

---

## 1. 프로젝트 소개

### 프로젝트명
OSS-Project

### 프로젝트 개요
이 프로젝트는 Python 기반으로 구현된 프로그램입니다.  
팀원들과 함께 GitHub를 활용하여 기능 개발, 버그 수정, 문서화, 브랜치 관리, 병합 과정을 수행합니다.

### 개발 목표
- Git/GitHub 협업 방식 익히기
- 브랜치 전략을 활용한 팀 개발 경험 쌓기
- Commit Convention 통일하기
- 프로젝트 문서화 및 유지보수성 향상

---

## 2. 개발 환경

- Language: Python 3
- Version Control: Git
- Collaboration: GitHub
- OS: macOS / Windows / Linux

---

## 3. 실행 방법

### 저장소 클론
```bash
git clone [저장소 주소]
cd OSS-Project
```

### 프로그램 실행
```bash
python3 main.py
```

---

## 4. 폴더 구조

```text
OSS-Project/
├── main.py
├── README.md
└── .gitignore
```

---

## 5. 브랜치 전략

본 프로젝트는 아래와 같은 브랜치 전략을 사용합니다.

- `main` : 최종 안정 버전
- `dev` : 개발 통합 브랜치
- `type/name/task` : 팀원별 작업 브랜치

### 브랜치 네이밍 규칙
- 형식: `type/name/task`
- 예시:
  - `docs/mincheol/readme`
  - `feat/jiyoon/user-input`
  - `fix/seojun/index-error`
  - `refactor/sohee/main-logic`

### 작업 흐름
1. `dev` 브랜치에서 최신 내용을 가져옵니다.
2. 각 팀원은 자신의 작업에 맞는 브랜치를 생성합니다.
3. 작업 완료 후 commit 및 push를 진행합니다.
4. 작업 브랜치를 `dev`에 merge합니다.
5. 작업이 끝난 브랜치는 삭제합니다.
6. 최종 점검 후 `dev`를 `main`에 merge합니다.

---

## 6. Commit Convention

commit message 형식은 아래와 같이 통일합니다.

```text
[파트] Type: 변경 내용
```

### 파트 구분
- `[AI]` : AI 관련 작업
- `[BE]` : 백엔드 관련 작업
- `[FE]` : 프론트엔드 관련 작업

### 사용 가능한 Type
- `Feat` : 새로운 기능 구현  
  예) 이미지 업로드, 알림 발송
- `Fix` : 버그 및 오류 수정
- `Design` : UI/UX 디자인 수정  
  예) React 컴포넌트 스타일링
- `AI` : 데이터셋 처리, 모델 학습, 성능 평가 관련 작업
- `Refactor` : 코드 로직 개선  
  기능 변화는 없지만 코드 품질 향상
- `Docs` : 문서 작성 및 수정  
  예) README, 회의록, API 명세서
- `Chore` : 빌드 설정, 패키지 설치, 환경 변수 설정 등 단순 작업
- `Test` : 테스트 코드 작성 및 검증

### 예시
```text
[BE] Feat: 회원정보 저장 기능 추가
[FE] Feat: 사용자 기간 입력 기능 추가
[BE] Feat: 사용자 로그인 기능 추가
[BE] Feat: 사용자 회원가입 기능 추가
[FE] Design: 메인 화면 버튼 스타일 수정
[AI] AI: 데이터 전처리 코드 추가
[BE] Fix: 로그인 예외 처리 오류 수정
[FE] Docs: README 실행 방법 수정
[BE] Refactor: 회원가입 함수 분리
[FE] Test: 입력 폼 검증 테스트 추가
[BE] Chore: requirements.txt 패키지 정리
```

---

## 7. 협업 규칙

- `main` 브랜치에는 직접 commit하지 않습니다.
- 모든 작업은 `dev` 브랜치에서 분기한 작업 브랜치에서 진행합니다.
- 작업 브랜치 이름은 `type/name/task` 형식을 따릅니다.
- commit 메시지는 `[파트] Type: 변경 내용` 형식을 따릅니다.
- merge 전 코드 및 변경 사항을 확인합니다.
- 의미 없는 commit 메시지(`수정`, `업데이트`, `test`)는 지양합니다.

---

## 8. 향후 계획

- [ ] 기본 기능 구현
- [ ] 예외 처리 추가
- [ ] 테스트 코드 작성
- [ ] README 보완
- [ ] 최종 발표 자료 정리

---

## 9. 라이선스

본 프로젝트는 공개소프트웨어 수업 목적의 프로젝트입니다.  
필요 시 적절한 오픈소스 라이선스를 추가할 수 있습니다.

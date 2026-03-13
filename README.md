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
- commit convention 통일하기
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
- `feature/기능이름` : 기능 개발 브랜치

### 작업 흐름 예시
1. `dev` 브랜치에서 `feature` 브랜치 생성
2. 기능 개발 후 commit
3. `dev` 브랜치로 merge
4. 검토 후 최종적으로 `main`에 반영

예시:
```bash
git checkout dev
git checkout -b feature/login
```

---

## 6. Commit Convention

commit message 형식은 아래와 같이 통일합니다.

```text
type: 변경 내용
```

### 사용 가능한 type
- `feat` : 새로운 기능 추가
- `fix` : 버그 수정
- `docs` : 문서 수정
- `style` : 코드 스타일 수정(기능 변화 없음)
- `refactor` : 리팩토링
- `test` : 테스트 코드 추가/수정
- `chore` : 설정, 폴더 정리, 기타 작업
- `init` : 초기 세팅

### 예시
```text
init: 프로젝트 초기 세팅
feat: 사용자 입력 기능 추가
fix: main.py 실행 오류 수정
docs: README 실행 방법 추가
refactor: 함수 분리로 코드 구조 개선
```

---

## 7. 협업 규칙

- `main` 브랜치에는 직접 commit하지 않습니다.
- 새로운 작업은 `feature` 브랜치에서 진행합니다.
- commit 메시지는 팀 convention을 따릅니다.
- merge 전 코드 및 변경 사항을 확인합니다.
- 의미 없는 commit 메시지(`수정`, `업데이트`, `test`)는 지양합니다.

---

## 8. 팀원 역할

| 이름 | 역할 | 담당 내용 |
|------|------|-----------|
| 팀원1 | 팀장 | 프로젝트 총괄, 일정 관리 |
| 팀원2 | 개발 | 기능 구현 |
| 팀원3 | 개발 | 기능 구현 및 테스트 |
| 팀원4 | 문서 | README, 발표 자료 정리 |

> 위 표는 팀 상황에 맞게 수정해서 사용하세요.

---

## 9. 향후 계획

- [ ] 기본 기능 구현
- [ ] 예외 처리 추가
- [ ] 테스트 코드 작성
- [ ] README 보완
- [ ] 최종 발표 자료 정리

---

## 10. 라이선스

본 프로젝트는 공개소프트웨어 수업 목적의 프로젝트입니다.  
필요 시 적절한 오픈소스 라이선스를 추가할 수 있습니다.

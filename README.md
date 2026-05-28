# 기타 이펙터 톤 매칭 & 연주 피드백 시스템

기준 음원과 사용자 연주 음원을 비교하여  
**이펙터 세팅 피드백**과 **연주 품질 피드백**을 JSON으로 반환하는 분석 파이프라인.

---

## 실행 방법

```bash
python main.py --ref <기준음원.wav> --copy <연주음원.wav>
```

### 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--ref` | 기준 음원 경로 (필수) | — |
| `--copy` | 비교 음원 경로 (필수) | — |
| `--active-effects` | 이펙터 목록 override (예: `dist delay`) | 파일명 자동 감지 |
| `--sr` | 샘플링 레이트 | `22050` |
| `--save` | 결과 JSON 저장 경로 | — |

### 실행 예시

```bash
python main.py --ref play/clean_solo_2.wav --copy play/drive_solo_2.wav
python main.py --ref ref.wav --copy user.wav --active-effects dist delay --save result.json
```

---

## 반환값

`run_unified_feedback()` 및 `main.py` 실행 결과는 아래 구조의 dict(JSON)를 반환한다.

```json
{
  "reference_path": "play/dist_100_solo_2.wav",
  "copy_path": "play/dist_25_solo_2.wav",
  "effect_feedback": {
    "overall_similarity": 0.85,
    "axes": [
      {
        "axis": "drive",
        "reference_amount": 80,
        "copy_amount": 30,
        "difference": -50,
        "similarity": 0.78,
        "action": "much_raise",
        "message": "[드라이브] 강도를 대폭 올려주세요! 🔼",
        "features": [
          {
            "feature": "crest_factor",
            "reference_value": 3.06,
            "copy_value": 5.36,
            "delta": 2.30,
            "percent_change": 75.0,
            "expected_direction": "decrease",
            "meaning": "clipping collapses peak-to-RMS ratio"
          }
        ]
      }
    ]
  },
  "playing_feedback": {
    "tone": {
      "overall_score": 82.5,
      "grade": "A",
      "issues": ["배음 구조가 원본보다 약합니다."],
      "suggestions": ["Presence 노브를 올려보세요."],
      "strengths": ["음정 정확도가 높습니다."]
    },
    "timeseries": {
      "pitch_score": 88.0,
      "rhythm_score": 72.0,
      "combined_score": 81.6,
      "grade": "A",
      "pitch_mae_semitone": 0.75,
      "onset_mae_ms": 95.0,
      "issues": ["박자가 원곡보다 밀리는 경향이 있습니다."],
      "suggestions": ["메트로놈에 맞춰 연습해보세요."],
      "strengths": ["음정 정확도가 우수합니다."]
    }
  }
}
```

### 최상위 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `reference_path` | string | 기준 음원 파일 경로 |
| `copy_path` | string | 비교 음원 파일 경로 |
| `effect_feedback` | object | 이펙터 세팅 피드백 |
| `playing_feedback` | object | 연주 품질 피드백 |

### effect_feedback

| 필드 | 타입 | 설명 |
|---|---|---|
| `overall_similarity` | float 0~1 | 세 축 전체 평균 유사도 |
| `axes` | array | drive / space / phase 세 축의 피드백 배열 |

**axes 원소**

| 필드 | 타입 | 설명 |
|---|---|---|
| `axis` | string | `"drive"` / `"space"` / `"phase"` |
| `reference_amount` | int 0~100 | 기준 음원의 이펙터 강도 추정값 |
| `copy_amount` | int 0~100 | 사용자 음원의 이펙터 강도 추정값 |
| `difference` | int | `copy_amount − reference_amount` |
| `similarity` | float 0~1 | 해당 축 유사도 |
| `action` | string | 아래 action 표 참고 |
| `message` | string | 사용자에게 보여줄 한국어 피드백 문장 |
| `features` | array | 판단 근거가 된 피처 목록 (keep이면 빈 배열) |

**features 원소**

| 필드 | 타입 | 설명 |
|---|---|---|
| `feature` | string | 피처명 (예: `crest_factor`, `zcr`) |
| `reference_value` | float | 기준 음원 피처값 |
| `copy_value` | float | 사용자 음원 피처값 |
| `delta` | float | `copy_value − reference_value` |
| `percent_change` | float | 변화율 (%) |
| `expected_direction` | string | `"increase"` / `"decrease"` — 이펙터가 강해질 때 이 피처가 향해야 할 방향 |
| `meaning` | string | 피처의 음향적 의미 설명 (영문) |

**action 값**

| action | 의미 |
|---|---|
| `keep` | 현재 강도 유지 |
| `raise` | 강도 소폭 올리기 |
| `much_raise` | 강도 대폭 올리기 |
| `lower` | 강도 소폭 낮추기 |
| `much_lower` | 강도 대폭 낮추기 |

### playing_feedback.tone

| 필드 | 타입 | 설명 |
|---|---|---|
| `overall_score` | float 0~100 | 음색 종합 점수 |
| `grade` | string | S / A / B / C / D |
| `issues` | array\<string\> | 문제점 목록 |
| `suggestions` | array\<string\> | 개선 제안 목록 |
| `strengths` | array\<string\> | 잘된 점 목록 |

### playing_feedback.timeseries

| 필드 | 타입 | 설명 |
|---|---|---|
| `pitch_score` | float 0~100 | 음정 정확도 점수 |
| `rhythm_score` | float 0~100 | 박자 정확도 점수 |
| `combined_score` | float 0~100 | pitch×0.6 + rhythm×0.4 종합 점수 |
| `grade` | string | S / A / B / C / D |
| `pitch_mae_semitone` | float | 평균 음정 오차 (반음 단위) |
| `onset_mae_ms` | float | 평균 타이밍 오차 (ms 단위) |
| `issues` | array\<string\> | 문제점 목록 |
| `suggestions` | array\<string\> | 개선 제안 목록 |
| `strengths` | array\<string\> | 잘된 점 목록 |

### 등급 기준

| grade | 톤 점수 | 시계열 점수 |
|---|---|---|
| S | 90점 이상 | 95점 이상 |
| A | 80점 이상 | 85점 이상 |
| B | 65점 이상 | 70점 이상 |
| C | 50점 이상 | 50점 이상 |
| D | 50점 미만 | 50점 미만 |

---

## 파일 구조

```
.
├── main.py                        # CLI 진입점
├── unified_pipeline.py            # 이펙터+연주 피드백 통합 함수
│
├── tone_match_model.py            # [이펙터 피드백] 핵심 모델
│
├── preprocessor.py                # [연주 피드백] 무음 제거 · RMS 정규화 · 길이 정렬
├── feature_extractor.py           # [연주 피드백] librosa 피처 추출
├── similarity.py                  # [연주 피드백] 코사인 유사도 계산
├── feedback.py                    # [연주 피드백] 톤 피드백 리포트 생성
│
├── pipeline_timeseries.py         # [시계열 피드백] 1~4단계 순차 실행
├── feature_extractor_timeseries.py# [시계열 피드백] 음정(pitch) · 온셋(onset) 추출
├── timeseries_align.py            # [시계열 피드백] DTW 정렬
├── timeseries_scoring.py          # [시계열 피드백] 오차 점수화
├── feedback_timeseries.py         # [시계열 피드백] 시계열 피드백 리포트 생성
│
└── random_pair_test.py            # 스모크 테스트 스크립트 (무작위 쌍 N회 실행)
```

---

## 파이프라인 흐름

```
ref.wav ─┐
          ├─▶ [이펙터 피드백] ToneMatchModel.compare()
copy.wav ─┘      drive / space / phase 축별 강도 비교

ref.wav ─┐
          ├─▶ [전처리] 무음 제거 → RMS 정규화 → 길이 정렬
copy.wav ─┘
               │
               ├─▶ [톤 분석] librosa 피처 → 코사인 유사도 → 피드백
               │
               └─▶ [시계열 분석] pitch/onset 추출 → DTW 정렬 → 오차 계산 → 피드백
```

---

## 이펙터 피드백 상세

`tone_match_model.py`의 `ToneMatchModel`이 세 축을 독립적으로 평가한다.

| 축 | 감지 이펙터 | 핵심 피처 |
|---|---|---|
| Drive | dist · overdrive · fuzz | crest_factor, zcr, rms, spectral_bandwidth, harmonic_ratio |
| Space | delay · reverb | energy_decay, spectral_flux_variance, sustain_ratio |
| Phase | phaser · chorus · tremolo | on/off만 판정 (강도 추정 없음) |

파일명에서 이펙터를 자동 감지하며, `active_effects` 인자로 override 가능.

---

## 연주 피드백 상세

### 톤 분석 (`similarity.py`)

| 피처 | 가중치 | 역할 |
|---|---|---|
| MFCC | 0.35 | 음색 / 공진 특성 |
| Spectral Contrast | 0.18 | 배음 구조 |
| BPM 유사도 | 0.10 | 템포 일치도 |
| Spectral Centroid | 0.10 | 음색 밝기 |
| Chroma | 0.08 | 음정 일치도 |
| Duration 유사도 | 0.07 | 연주 길이 |
| Tonnetz | 0.05 | 조성 공간 |
| Spectral Bandwidth | 0.03 | 음색 넓이 |
| Spectral Rolloff | 0.02 | 고음 분포 |
| ZCR / RMS | 0.01×2 | 거칠기 / 다이나믹 |

코사인 유사도에 0.75 threshold stretch 적용 → 변별력 확보.

### 시계열 분석 (`pipeline_timeseries.py`)

1. **피처 추출**: crepe 기반 pitch MIDI + librosa onset 시계열
2. **DTW 정렬**: 연주 속도 차이를 보정하여 프레임 단위 매핑
3. **오차 계산**: pitch MAE (반음) + onset MAE (ms)
4. **점수화**: pitch 60% + rhythm 40% 가중 합산

---

## 평가 결과 (이펙터 모델)

200쌍 무작위 샘플링 (seed=42):

| 축 | 정확도 |
|---|---|
| Drive (dist) | 75% |
| Space (delay) | 87% |
| Phase (phaser) | 100% (on/off 판정) |
| **전체** | **66%** |

| 이펙터 조합 | 정확도 |
|---|---|
| 단독 | ~95% |
| 2개 조합 | ~75% |
| 3개 동시 | ~63% |

3개 이펙터 동시 적용 시 cross-axis 오염으로 정확도 하락.  
실제 사용 환경(이펙터 하나씩 조정)에서는 더 높은 정확도 기대.

---

## 의존성

```
librosa
numpy
scikit-learn
crepe
```

```bash
pip install -r requirements.txt
```

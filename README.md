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
        "reference_amount": 50,
        "copy_amount": 73,
        "difference": 23,
        "similarity": 0.77,
        "action": "lower",
        "message": "[드라이브] 강도를 살짝 줄여주세요. 🔽",
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
    "pitch_score": 88.0,
    "rhythm_score": 72.0,
    "combined_score": 81.6,
    "grade": "A",
    "pitch_reliable": true,
    "pitch_mae_semitone": 0.75,
    "onset_mae_ms": 95.0,
    "issues": ["박자 점수 72.0점: 전체적으로 박자가 원곡보다 느리게 밀리는 경향이 있습니다."],
    "suggestions": ["메트로놈에 맞춰 연습해 보세요."],
    "strengths": ["음정 정확도가 우수합니다."]
  }
}
```

### 최상위 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `reference_path` | string | 기준 음원 파일 경로 |
| `copy_path` | string | 비교 음원 파일 경로 |
| `effect_feedback` | object | 이펙터 세팅 피드백 |
| `playing_feedback` | object | 연주 품질 피드백 (음정 · 박자) |

### effect_feedback

| 필드 | 타입 | 설명 |
|---|---|---|
| `overall_similarity` | float 0~1 | 세 축 전체 평균 유사도 |
| `axes` | array | drive / space / phase 세 축의 피드백 배열 |

**axes 원소**

| 필드 | 타입 | 설명 |
|---|---|---|
| `axis` | string | `"drive"` / `"space"` / `"phase"` |
| `reference_amount` | int | 항상 50 (기준점 고정값) |
| `copy_amount` | int 0~100 | `50 + difference`로 계산한 사용자 강도 표현값 |
| `difference` | int -100~+100 | 양수 = copy가 ref보다 이펙터 강함, 음수 = 약함 |
| `similarity` | float 0~1 | 해당 축 유사도 (`1 - \|difference\| / 100`) |
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

### playing_feedback

연주 피드백은 음정(pitch)과 박자(rhythm) 시계열 분석만 담당한다.  
음색(tone) 분석은 이펙터 피드백(effect_feedback)과 중복되므로 제외했다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `pitch_score` | float 0~100 | 음정 정확도 점수 |
| `rhythm_score` | float 0~100 | 박자 정확도 점수 |
| `combined_score` | float 0~100 | 종합 점수 (아래 참고) |
| `grade` | string | S / A / B / C / D |
| `pitch_reliable` | bool | `false`면 딜레이 이펙터가 감지되어 음정 점수 신뢰도 낮음 |
| `pitch_mae_semitone` | float | 평균 음정 오차 (반음 단위) |
| `onset_mae_ms` | float | 평균 타이밍 오차 (ms 단위) |
| `issues` | array\<string\> | 진단 메시지 목록 |
| `suggestions` | array\<string\> | 개선 제안 목록 |
| `strengths` | array\<string\> | 잘된 점 목록 |

**combined_score 계산 방식**

```
pitch_reliable = true  → combined = pitch × 0.6 + rhythm × 0.4
pitch_reliable = false → combined = rhythm (음정 점수 제외)
```

딜레이 잔향이 음정 추적을 교란할 수 있으므로, 딜레이가 감지된 경우 부정확한 음정 점수를 종합 점수에서 제외하여 오판을 방지한다.

### 등급 기준

| grade | combined_score |
|---|---|
| S | 95점 이상 |
| A | 85점 이상 |
| B | 70점 이상 |
| C | 50점 이상 |
| D | 50점 미만 |

---

## 파일 구조

```
.
├── main.py                          # CLI 진입점
├── unified_pipeline.py              # 이펙터+연주 피드백 통합 함수
│
├── tone_match_model.py              # [이펙터 피드백] 핵심 모델
│
├── pipeline_timeseries.py           # [연주 피드백] 1~4단계 순차 실행
├── feature_extractor_timeseries.py  # [연주 피드백] 음정(pitch) · 온셋(onset) 추출
├── timeseries_align.py              # [연주 피드백] DTW 정렬
├── timeseries_scoring.py            # [연주 피드백] 오차 점수화
├── feedback_timeseries.py           # [연주 피드백] 피드백 리포트 생성
│
└── random_pair_test.py              # 스모크 테스트 스크립트 (무작위 쌍 N회 실행)
```

---

## 파이프라인 흐름

```
ref.wav ─┐
          ├─▶ [이펙터 피드백] ToneMatchModel.compare()
copy.wav ─┘      drive / space / phase 축별 강도 비교

ref.wav ─┐
          ├─▶ [연주 피드백] run_timeseries_pipeline()
copy.wav ─┘      1. 피처 추출 (pyin 음정 + onset 박자 + silence_ratio)
                 2. DTW 정렬 (연주 속도 차이 보정)
                 3. 오차 계산 (pitch MAE · onset MAE)
                 4. 피드백 생성 (점수 · 등급 · 메시지)
```

---

## 이펙터 피드백 상세

`tone_match_model.py`의 `ToneMatchModel`이 세 축을 독립적으로 평가한다.

| 축 | 감지 이펙터 | 핵심 피처 |
|---|---|---|
| Drive | dist | crest_factor, rms, zcr, spectral_bandwidth, harmonic_ratio |
| Space | delay | energy_decay, spectral_flux_variance, sustain_ratio |
| Phase | phaser | on/off만 판정 (강도 추정 없음, 항상 keep 반환) |

파일명에서 이펙터를 자동 감지하며(`dist_50_delay_100_solo_3.wav` → `["dist", "delay"]`), `--active-effects` 인자로 override 가능.  
`chorus`, `reverb` 등 분석 범위 밖의 이펙터는 파일명에 있어도 무시한다.

### 점수 계산 방식

각 축은 피처별 delta를 tanh로 압축한 뒤 가중 평균을 낸다.

```
signed_delta = (copy - ref) / |ref|        # ref 대비 변화 비율 (direction 반영)
contribution = tanh(signed_delta × sensitivity)   ∈ [-1, 1]
axis_delta   = Σ(contribution × weight) / Σ(weight)
difference   = round(axis_delta × 100)     ∈ [-100, 100]
```

tanh를 사용하는 이유는 특정 피처 하나가 극단값을 가질 때 점수를 독점하지 않도록 압축하기 위해서다.

---

## 연주 피드백 상세

### 시계열 분석 (`pipeline_timeseries.py`)

1. **피처 추출**: pyin 기반 pitch MIDI + librosa onset 시계열 + silence_ratio(무음 비율)
2. **delay 감지**: silence_ratio < 0.14이면 딜레이 감지 → `pitch_reliable = false`
3. **DTW 정렬**: 연주 속도 차이를 보정하여 음 대 음으로 매핑
4. **오차 계산**: pitch MAE (반음) + onset MAE (ms)
5. **점수화**: pitch 60% + rhythm 40% (딜레이 감지 시 rhythm 100%)

### 점수 기준

| 항목 | 만점 기준 | 0점 기준 |
|---|---|---|
| 음정 | MAE ≤ 0.5 반음 | MAE ≥ 6.0 반음 |
| 박자 | MAE ≤ 30 ms | MAE ≥ 400 ms |

### delay 신뢰도 판정

딜레이 이펙터는 노트 사이 무음 구간을 잔향으로 채우기 때문에 pyin의 음정 추적을 교란한다.  
실험 결과 무음 비율이 딜레이 없음 16~20%, 딜레이 있음 7~12%로 분리됨을 확인하여 임계값 0.14를 채택했다.

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
scipy
dtaidistance
```

```bash
pip install -r requirements.txt
```

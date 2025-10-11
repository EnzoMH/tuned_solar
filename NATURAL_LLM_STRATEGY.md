# 자연스럽게 대답하는 LLM 만들기 전략

## 목표

**하이퍼파라미터 조정 없이** 자연스러운 응답을 생성하는 LLM

```python
# 이상적인 목표
response = model.generate(
    inputs,
    max_new_tokens=512,
    # temperature, top_p, repetition_penalty 같은 조정 없이
    # 기본값만으로 자연스러운 응답!
)
```

---

### 1. 데이터 품질 > 데이터 양 

#### 좋은 데이터의 조건

**A. 일관된 톤과 스타일**
```json
✅ 좋은 예:
{
  "user": "안녕하세요",
  "assistant": "안녕하세요! 무엇을 도와드릴까요?"
}
{
  "user": "날씨 알려줘",
  "assistant": "죄송합니다. 실시간 날씨 정보는 제공할 수 없습니다."
}

❌ 나쁜 예 (톤 불일치):
{
  "user": "안녕하세요",
  "assistant": "안녕! 뭐 필요해?" ← 반말/존댓말 혼재
}
```

**B. 적절한 길이**
```
질문: 15-500자
답변: 50-1500자

너무 짧음 (❌): "네"
너무 김 (❌): 2000자 이상의 장황한 설명
적절 (✅): 150-500자의 명확한 답변
```

**C. 반복 없는 답변**
```
❌ 나쁜 예:
"안녕하세요. 안녕하세요. 제가 도와드리겠습니다. 도와드리겠습니다."

✅ 좋은 예:
"안녕하세요! 무엇을 도와드릴까요?"
```

---

### 2. 과적합 방지 (일반화 능력) ⭐⭐⭐

#### 현재 설정 (최적화됨 ✅)
```python
num_train_epochs: int = 2  # ✅ 191K 데이터면 2-3 에포크 충분
lora_dropout: float = 0.05  # ✅ 낮은 dropout (0.05-0.1)
weight_decay: float = 0.01  # ✅ 정규화
```

#### 과적합 징후 확인
```
Epoch 1: train_loss=1.5, eval_loss=1.6 ✅ 정상
Epoch 2: train_loss=1.2, eval_loss=1.3 ✅ 정상
Epoch 3: train_loss=0.8, eval_loss=1.1 ⚠️ 주의
Epoch 5: train_loss=0.3, eval_loss=1.5 ❌ 과적합!
```

**대책**: 
- Eval loss가 증가하기 시작하면 **즉시 중단**
- Early stopping 활용
- 최고 체크포인트 사용 (final 아님)

---

### 3. 레이블 마스킹 (필수!) ⭐⭐⭐

#### 현재 적용 상태 ✅
```python
# solar_v2.py (이미 적용됨)
labels = input_ids.clone()
labels[:prompt_length] = -100  # 질문 부분 마스킹
labels[labels == pad_token_id] = -100  # padding 마스킹
```

**효과**:
- ✅ 답변 생성만 학습 (질문 생성 안 함)
- ✅ 혼란 없는 명확한 학습
- ✅ 자연스러운 응답 패턴

---

### 4. 포맷 일관성 (필수!) ⭐⭐⭐

#### 현재 설정 ✅
```python
# 훈련 시
prompt = f"### User:\n{user}\n\n### Assistant:\n"

# 추론 시 (conv.py도 동일하게)
prompt = f"### User:\n{user}\n\n### Assistant:\n"
```

**중요**: 훈련과 추론의 **완벽한 일치**가 자연스러운 응답의 핵심!

---

### 5. 적절한 학습률과 스케줄 ⭐⭐

#### 현재 설정
```python
learning_rate: float = 2e-4  # ✅ 적절
warmup_ratio: float = 0.03   # ✅ 3% warmup
```

**더 자연스럽게 하려면**:
```python
learning_rate: float = 1e-4  # 2e-4 → 1e-4 (더 보수적)
warmup_ratio: float = 0.05   # 0.03 → 0.05 (더 안정적)
```

---

## 🎨 자연스러운 응답을 위한 고급 기법

### A. DPO (Direct Preference Optimization) - 나중 단계

```python
# 1차: SFT (지금 하는 것)
model = train_sft(data)

# 2차: DPO (선호도 학습)
# "좋은 답변"과 "나쁜 답변"을 비교 학습
preference_data = [
    {
        "prompt": "안녕하세요",
        "chosen": "안녕하세요! 무엇을 도와드릴까요?",  # 선호
        "rejected": "ㅎㅇㅎㅇ 뭐함?"  # 거부
    }
]
```

**효과**: 
- 더 자연스럽고 적절한 답변
- temperature 조정 없이도 좋은 품질

---

### B. 데이터 다양성 vs 일관성 밸런스

```python
# 현재 데이터 분포
smol_koreantalk: 88,752개 (46.3%) - 대화
kowiki_qa: 48,699개 (25.4%) - 지식
kullm_v2: 33,422개 (17.4%) - Instruction
koalpaca: 20,768개 (10.8%) - Instruction

# 문제: 대화 데이터가 46%로 너무 많음
# → 캐주얼한 톤이 될 수 있음
```

**개선**:
```python
# 비율 조정 (instruction 중심으로)
max_samples: Optional[int] = 100000

# 또는 소스별 가중치
- Instruction: 60%
- 대화: 25%
- 지식: 15%
```

---

### C. 샘플링 전략

```python
# 데이터가 많으면 샘플링
max_samples: Optional[int] = 100000  # 191K → 100K
```

**장점**:
- ✅ 훈련 시간 단축 (155시간 → 80시간)
- ✅ 과적합 위험 감소
- ✅ 고품질 샘플 선별 가능

---

## 📊 최종 권장 설정

### solar_v2.py (자연스러운 LLM 최적화)

```python
@dataclass
class SOLARFineTuningConfig:
    # 모델
    base_model: str = "upstage/SOLAR-10.7B-Instruct-v1.0"  # ✅ 원래 모델
    
    # 데이터
    data_path: str = ".../korean_large_dataset.json"
    max_samples: Optional[int] = 100000  # ✅ 100K로 샘플링
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05  # ✅ 낮은 dropout
    
    # 훈련
    num_train_epochs: int = 2  # ✅ 과적합 방지
    per_device_train_batch_size: int = 8  # ✅ 효율적
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4  # ✅ 보수적 (선택)
    warmup_ratio: float = 0.05  # ✅ 안정적 (선택)
```

**예상 결과**:
- 훈련 시간: **24-28시간** (현실적!)
- 한국어 능력: ★★★★☆
- 자연스러움: ★★★★★
- 안정성: ★★★★★

---

## 🎬 추론 시 기본 설정 (자연스러운 응답)

### conv.py 권장 설정

```python
def generate_response(self, user_input, 
                      max_new_tokens=256,  # ← 적절한 길이
                      temperature=0.7,     # ← 약간의 다양성
                      top_p=0.9,           # ← 적절한 sampling
                      repetition_penalty=1.1):  # ← 낮은 패널티
```

**더 자연스럽게**:
```python
# temperature만 낮춰도 충분
temperature=0.3  # 일관성 있는 답변
top_p=0.85
repetition_penalty=1.0  # 필요 없을 수도!
```

---

## 🔬 검증 방법

### 체크포인트별 테스트

```bash
# 각 체크포인트를 기본 설정으로 테스트
python conv.py --model-path .../checkpoint-250

# 다양한 질문으로 테스트
사용자: 안녕하세요
사용자: 피보나치 수열이 뭐야?
사용자: 파이썬 코드 작성해줘
사용자: 오늘 날씨 어때?
```

**좋은 징후**:
- ✅ 간결하고 명확한 답변
- ✅ 반복 없음
- ✅ 적절한 길이
- ✅ 일관된 톤

**나쁜 징후**:
- ❌ 너무 길거나 짧음
- ❌ 반복 (ㅋㅋㅋ, 같은 문장 반복)
- ❌ 톤 불일치
- ❌ 이상한 문자

---

## 🚀 즉시 실행 가능한 최적화 버전

추가로 샘플링 설정을 추가하겠습니다:

<function_calls>
<invoke name="search_replace">
<parameter name="file_path">/home/work/tesseract/solar_v2.py

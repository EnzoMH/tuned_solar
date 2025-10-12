# WMS Instruction Tuning Dataset Generator

WMS 도메인 특화 LLM 파인튜닝을 위한 고품질 QA 데이터셋 자동 생성 도구

## 프로젝트 구조

```
Instruction/
├── README.md                    # 이 파일
├── PERSONA_README.md            # 페르소나 상세 가이드
├── requirements.txt             # 의존성 패키지
│
├── gen_qa_dtset.py              # QA 데이터셋 생성 스크립트 (메인 실행)
├── vectordb_builder.py          # FAISS 벡터 DB 구축
│
├── conv_gen/                    # 대화형 데이터 생성 모듈
│   ├── q_gen.py                 # EXAONE 질문 생성기
│   ├── a_maker.py               # EEVE 답변 생성기 (RAG)
│   ├── qa_pipeline.py           # Q-A 통합 파이프라인 (고급)
│   └── personas.py              # 페르소나 정의
│
└── output/                      # 생성된 데이터셋
```

## 역할 분담

### 현재 구성
- **EXAONE 4.0 1.2B**: 얕은 사전조사를 한 실무자 역할로 질문 생성 (FAISS 미사용, 자체 지식만)
- **EEVE (MyeongHo0621/eeve-vss-smh) + FAISS**: 10년 경력 전문가 역할로 답변 생성 (FAISS RAG 기반)

### 지식 격차의 효과
- **질문자 (EXAONE)**: 얕은 지식 → 현실적이고 솔직한 질문
- **답변자 (EEVE)**: 깊은 지식 → 전문적이고 실무적인 답변
- **결과**: 자연스러운 Q-A 대화 흐름

## Quick Start

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# EXAONE과 EEVE 모델 자동 다운로드됨 (GPU 필요)
```

### 2. FAISS 벡터 DB 준비

프로젝트 루트에 `faiss_storage` 디렉토리가 있어야 합니다:

```
tesseract/
├── faiss_storage/
│   ├── index.faiss (또는 warehouse_automation_knowledge.index)
│   └── index.pkl
└── datageneration/Instruction/
```

### 3. 실행 (간단한 방법)

```bash
# 방법 1: 수동 질문 리스트 + EEVE 답변 (추천, 가장 빠름)
python gen_qa_dtset.py --mode manual

# 방법 2: EXAONE 질문 생성 + EEVE 답변
python gen_qa_dtset.py --mode exaone --num-topics 2 --num-personas 2 --questions-per-topic 2

# 출력: output/wms_qa_dataset_[timestamp].json
```

### 3-1. 고급 사용 (개별 모듈)

```bash
cd conv_gen

# 질문만 생성 (EXAONE)
python q_gen.py

# 답변만 생성 (질문 리스트 필요)
python a_maker.py

# 전체 파이프라인 (커스터마이징 가능)
python qa_pipeline.py
```

## 생성 프로세스

### `gen_qa_dtset.py` 사용 (추천)

#### 방법 1: 수동 질문 + EEVE 답변 (가장 빠름)
```bash
python gen_qa_dtset.py --mode manual
```
1. 미리 정의된 5개의 현실적인 질문 사용
2. EEVE가 각 질문에 대해 FAISS 검색 후 답변 생성
3. JSON으로 저장

#### 방법 2: EXAONE 질문 + EEVE 답변 (자동화)
```bash
python gen_qa_dtset.py --mode exaone --num-topics 2 --num-personas 2 --questions-per-topic 2
```
1. **EXAONE**이 자체 지식만으로 페르소나 기반 질문 생성 (얕은 사전조사)
2. **EEVE**가 각 질문에 대해 FAISS 검색 후 답변 생성 (깊은 전문 지식)
3. 지식 격차로 더 자연스러운 Q-A 생성
4. JSON으로 저장

## 개별 모듈 사용법 (Python API)

### 질문 생성만 (EXAONE)

```python
from conv_gen.q_gen import ExaoneQuestionMaker, WMS_TOPICS, PERSONAS

maker = ExaoneQuestionMaker()
questions = maker.generate_diverse_questions(
    topics=WMS_TOPICS[:2],
    personas=PERSONAS[:2],
    questions_per_topic=2
)
maker.cleanup()
```

### 답변 생성만 (EEVE + FAISS)

```python
from conv_gen.a_maker import AnswerMaker

maker = AnswerMaker()
result = maker.generate_answer(
    question="WMS 도입 비용이 얼마나 드나요?"
)
print(result['answer'])
```

### 전체 파이프라인 (고급)

```python
from conv_gen.qa_pipeline import QAPipeline

pipeline = QAPipeline()
result = pipeline.generate_answers_from_questions(
    questions=["질문1", "질문2", "질문3"]
)
```

## 설정 커스터마이징

### EEVE 답변 생성 파라미터 (a_maker.py)

답변 품질을 조정하려면 `a_maker.py`의 `generate_answer()` 메서드에서 다음 파라미터를 수정하세요:

```python
outputs = self.model.generate(
    max_new_tokens=512,           # 답변 최대 길이
    temperature=0.3,              # 낮을수록 일관성↑, 높을수록 창의성↑
    top_p=0.85,                   # Nucleus sampling
    repetition_penalty=1.15,      # 반복 방지 (1.0~1.3 권장)
    do_sample=True
)
```

**repetition_penalty 권장값:**
- `1.0`: 반복 제한 없음 (자연스럽지만 반복 가능)
- `1.15`: **최적 균형** (반복 감소 + 자연스러움 유지)
- `1.3+`: 과도한 제한 (부자연스러운 문장)

### Gemini 질문 생성 파라미터 (q_gen.py)

```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.8,              # 질문 다양성
    max_output_tokens=800
)
```

### 페르소나 커스터마이징

`conv_gen/personas.py` 또는 `conv_gen/q_gen.py`의 `PERSONAS`와 `WMS_TOPICS`를 수정하여 질문 스타일을 변경할 수 있습니다.

자세한 내용은 `PERSONA_README.md`를 참고하세요.

## 데이터 형식

`gen_qa_dtset.py`가 생성하는 JSON 형식 (LLaMA 3.1/3.2 호환):

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "당신은 10년 이상의 경력을 가진 물류 시스템 전문가입니다. WMS(창고관리시스템) 도입과 운영에 대한 실무 경험을 바탕으로 상담을 제공합니다."
      },
      {
        "role": "user",
        "content": "재고 실사하는데 하루 종일 걸리는데, 얼마나 줄일 수 있나요?"
      },
      {
        "role": "assistant",
        "content": "실제 제가 진행한 중소 물류센터 프로젝트를 보면요, 재고 실사 시간이 하루 8시간에서 2시간으로 줄었습니다. 다만 처음 3개월은 직원들이 적응하느라 오히려 더 느렸어요..."
      }
    ],
    "metadata": {
      "id": 1,
      "topic": "재고 정확도 향상 방법",
      "persona": "김영수 (중소 물류센터 관리자)",
      "num_contexts": 5,
      "inference_time": 3.42,
      "output_tokens": 256
    }
  }
]
```

**LLaMA 3.1/3.2 Instruction Tuning 직접 호환 형식**

## 주요 특징

### 답변 품질 (EEVE + FAISS)
- **깊은 전문 지식**: FAISS RAG 기반 10년 경력 전문가의 답변
- **실무 전문가 톤**: "참고자료에 따르면" ❌ → "실제 제가 진행한 프로젝트에서는" ✅
- **구체적 수치 포함**: "평균 3-6개월", "약 30% 절감", "하루 500건 → 1,200건"
- **솔직한 장단점 언급**: 좋은 점만 나열하지 않고 주의사항도 포함
- **자연스러운 대화체**: 상담하듯이 설명
- **repetition_penalty 1.15**: 반복 감소 + 자연스러움 최적 균형

### 질문 품질 (EXAONE)
- **얕은 사전조사 페르소나**: FAISS 없이 자체 지식만으로 질문 생성
- **5가지 페르소나**: 중소 물류센터, 이커머스, 3PL, 제조업, 스타트업
- **현실적인 고민**: "50대 직원들도 배울 수 있나요?", "엑셀 데이터 옮길 수 있나요?"
- **수동 모드 대안**: 빠른 테스트를 위한 미리 정의된 질문

### 사용 편의성
- **간단한 실행**: `python gen_qa_dtset.py --mode manual` (수동) 또는 `--mode exaone` (자동)
- **즉시 사용 가능**: 설정 없이 바로 QA 데이터셋 생성
- **LLaMA 3.1/3.2 호환**: messages 형식으로 바로 Instruction Tuning 가능
- **system 프롬프트 내장**: "10년 경력 물류 전문가" 페르소나 자동 포함
- **메모리 관리**: 자동 cleanup으로 GPU 메모리 효율적 사용

## 다음 단계: LLaMA Instruction Tuning

생성된 데이터셋은 LLaMA 3.1/3.2 Instruction Tuning에 바로 사용 가능합니다:

```bash
# 생성된 데이터셋 확인
ls output/wms_qa_dataset_*.json

# LLaMA 파인튜닝 (예시)
# - Unsloth, Axolotl, TRL 등 어떤 도구든 사용 가능
# - messages 형식을 자동으로 인식
```

**데이터셋 특징:**
- ✅ system 프롬프트로 전문가 페르소나 정의
- ✅ user 메시지로 현실적인 질문
- ✅ assistant 메시지로 실무 경험 기반 답변
- ✅ metadata로 추적 가능한 생성 정보

## 문의 및 기여

Issues와 PR은 언제든 환영합니다!

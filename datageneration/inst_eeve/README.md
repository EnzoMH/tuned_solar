# EEVE 기반 WMS Instruction 데이터 생성기

EEVE-Korean-Instruct 모델을 사용한 WMS 도메인 특화 instruction 데이터 생성 도구

## architect Structure

```
inst_eeve/
├── README.md              # 이 파일
├── ans_gen_ev.py         # EEVE 기반 WMS 답변 생성기
└── (향후 추가될 파일들)
```

## 목적

1. **템플릿 일관성**: 1단계 파인튜닝과 동일한 EEVE 프롬프트 템플릿 사용
2. **RAG 통합**: FAISS VectorDB와 통합하여 참고 자료 기반 답변 생성
3. **WMS 특화**: WMS 도메인에 최적화된 instruction 데이터 생성

## 사용 방법

### 1. depedencies isntall 

```bash
pip install torch transformers peft accelerate
```

### 2. TODO 확인 및 설정

`ans_gen_ev.py` 파일의 TODO 주석을 확인하고 경로를 설정하세요:

```python
# TODO: 베이스 모델 확인
base_model_path = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

# TODO: 파인튜닝 완료 후 경로 설정 (옵션)
adapter_path = None  # 또는 "/home/work/eeve-korean-output/final"

# TODO: FAISS VectorDB 로더 경로 확인
from vectordb_loader import FAISSVectorDBLoader
```

### 3. 테스트 실행

```bash
cd /home/work/tesseract/datageneration/inst_eeve

# 단독 테스트
python ans_gen_ev.py
```

### 4. 실제 사용

```python
from ans_gen_ev import EEVEAnswerGenerator, PromptTemplate, GenerationConfig
from vectordb_loader import FAISSVectorDBLoader

# VectorDB 로드
vectordb = FAISSVectorDBLoader()

# 생성기 초기화
generator = EEVEAnswerGenerator(
    vectordb_loader=vectordb,
    base_model_path="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    adapter_path="/home/work/eeve-korean-output/final",  # 1단계 완료 후
    use_4bit=True
)

# 답변 생성
result = generator.generate_answer(
    question="WMS 도입 비용은 얼마나 드나요?",
    persona_name="김영수",
    persona_background="중소 물류센터 관리자",
    use_rag=True
)

print(result['answer'])
```

## ⚙️ 설정 커스터마이징

### GenerationConfig 클래스

```python
class GenerationConfig:
    MAX_TOKENS = 600              # TODO: 최대 답변 길이 조정
    TEMPERATURE = 0.8             # TODO: 다양성 조정 (0.5-1.0)
    TOP_P = 0.9
    TOP_K = 50
    REPETITION_PENALTY = 1.15
    
    # RAG 설정
    TOP_K_DOCS = 3                # TODO: 검색 문서 수
    MAX_CONTEXT_LENGTH = 1500     # TODO: 컨텍스트 길이
```

### PromptTemplate 클래스

통합 프롬프트 템플릿 (파인튜닝과 일관성 유지):

- `wms_with_context()`: RAG 컨텍스트 포함
- `wms_general()`: 일반 WMS 질문

## 🔄 워크플로우

### 1단계: 일반 한국어 Instruction 파인튜닝 (완료)

```bash
# eeve_finetune.py 사용
nohup python eeve_finetune.py > training_eeve.log 2>&1 &
```

→ 결과: `/home/work/eeve-korean-output/final`

### 2단계: WMS 데이터 생성 (현재 단계)

```bash
# ans_gen_ev.py 사용
python ans_gen_ev.py
```

→ 결과: WMS 도메인 instruction 데이터셋

### 3단계: WMS 특화 파인튜닝 (향후)

1단계 모델 + WMS 데이터로 추가 파인튜닝

## 📝 TODO 체크리스트

**ans_gen_ev.py 설정:**
- [ ] 베이스 모델 경로 확인 (`base_model_path`)
- [ ] 어댑터 경로 설정 (`adapter_path`, 1단계 완료 후)
- [ ] FAISS VectorDB 경로 확인
- [ ] 테스트 질문 수정 (실제 WMS 질문)
- [ ] 생성 파라미터 조정 (`GenerationConfig`)

**테스트:**
- [ ] 단독 실행 테스트 (`python ans_gen_ev.py`)
- [ ] RAG 검색 테스트
- [ ] 생성 품질 확인
- [ ] 메모리 사용량 확인

## 🆚 기존 Instruction/ 디렉토리와 차이점

| 항목 | 기존 (Instruction/) | 신규 (inst_eeve/) |
|------|-------------------|------------------|
| 모델 | SOLAR + LoRA | EEVE (1단계 파인튜닝) |
| 템플릿 | 커스텀 | EEVE 공식 (일관성) |
| 용도 | 초기 데이터 생성 | 2단계 WMS 특화 |
| 프롬프트 | 한국어 직접 지시 | EEVE 표준 템플릿 |

## 📊 기대 효과

1. **템플릿 일관성**: 1단계와 동일한 템플릿으로 성능 저하 방지
2. **품질 향상**: 1단계 파인튜닝된 모델 사용으로 더 자연스러운 한국어
3. **도메인 특화**: RAG를 통한 정확한 WMS 정보 반영
4. **반말→존댓말**: 1단계에서 학습한 정중한 답변 스타일 유지

## 🔗 관련 파일

- 상위 디렉토리: `../Instruction/` (기존 SOLAR 기반 생성기)
- 파인튜닝 스크립트: `../../eeve_finetune.py`
- 대화 테스트: `../../conv_eeve.py`
- 모델 병합: `../../merge_lora_to_base.py`

## 📚 참고

- EEVE 모델: https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0
- PEFT (LoRA): https://github.com/huggingface/peft


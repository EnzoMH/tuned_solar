# WMS Instruction Tuning Dataset Generation Pipeline

WMS 도메인 특화 LLM 파인튜닝을 위한 고품질 QA 데이터셋 자동 생성 파이프라인

---

## 🎯 프로젝트 목표

크롤링한 WMS 문서를 활용하여 **Instruction Tuning용 QA 데이터셋을 자동으로 생성**하고, 이를 통해 SOLAR 모델을 WMS 도메인 전문가로 파인튜닝합니다.

---

## 🏗️ 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     1. 데이터 수집 단계                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
        WMS 문서 크롤링 (.txt, .csv, .json, .pdf)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  2. 벡터 DB 구축 단계                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
    Langchain + FAISS + Korean Embeddings
    (jhgan/ko-sroberta-multitask)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               3. QA 데이터셋 생성 단계 (핵심)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────┴──────────────────┐
        │                                     │
   [Gemini 1.5 Flash]              [SOLAR + FAISS]
   물류 현업자 역할                  WMS 전문가 역할
   질문 생성 (VectorDB X)          답변 생성 (VectorDB O)
        │                                     │
        └──────────────────┬──────────────────┘
                            ↓
              Instruction Tuning Dataset
              (JSON format)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  4. 파인튜닝 단계                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
    SOLAR-10.7B + LoRA (QLoRA)
    KT Cloud H100E 환경
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  5. 배포 단계                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
    모델 병합 → 양자화 (FP8) → vLLM/SGLang → FastAPI
```

---

## 🎭 역할 기반 접근법 (Role-based Approach)

### Gemini = 물류 현업자 (Question Generator)

**역할**: WMS 도입을 고민하는 다양한 현업자
**VectorDB 사용**: ❌ (순수하게 역할 연기로 질문 생성)

**5가지 페르소나**:
1. **김영수** - 중소 물류센터 관리자
   - 관심사: 비용, 직원 교육, 기존 시스템 호환
   - 질문 스타일: 실용적, 직설적

2. **박지현** - 대형 이커머스 물류 책임자
   - 관심사: 처리 속도, 오류율, 피크 시즌 대응
   - 질문 스타일: 데이터 중심, 구체적

3. **이민호** - 3PL 업체 운영팀장
   - 관심사: 유연성, 다중 고객 관리, 보고서
   - 질문 스타일: 시스템 기능 중심

4. **최수진** - 제조업 물류 담당자
   - 관심사: ERP 연동, 재고 정확도, 생산 연계
   - 질문 스타일: 통합 프로세스 질문

5. **정태윤** - 스타트업 창고 운영자
   - 관심사: 초기 비용, 확장성, 사용 편의성
   - 질문 스타일: 초보자 기본 질문

### SOLAR = WMS 전문가 (Answer Generator)

**역할**: WMS 시스템 전문 컨설턴트
**VectorDB 사용**: ✅ (FAISS 검색 → 문서 기반 답변)

**답변 프로세스**:
1. 질문 받음
2. FAISS에서 관련 문서 검색 (top-k=3)
3. 검색된 문서 기반으로 답변 생성
4. 출처 정보 포함

---

## 📦 디렉토리 구조

```
wms-qa-pipeline/
├── README.md                          # 이 파일
├── .env                               # API Keys
├── requirements.txt                   # 의존성 패키지
│
├── example.py                         # 메인 실행 파일
│
├── data/
│   ├── crawled/                       # 크롤링한 원본 데이터
│   │   ├── *.txt
│   │   ├── *.csv
│   │   └── *.json
│   │
│   ├── vectordb/                      # FAISS 벡터 DB
│   │   └── wms_vectordb/
│   │
│   └── output/                        # 생성된 데이터셋
│       └── wms_instruction_dataset.json
│
├── models/
│   ├── solar-korean-wms/             # 기존 파인튜닝 모델 (LoRA)
│   └── solar-korean-wms-v2/          # 추가 파인튜닝 모델
│
└── training/
    └── solar_finetuning.py           # 파인튜닝 스크립트
```

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.0.38
faiss-cpu>=1.8.0
sentence-transformers>=2.5.0
python-dotenv>=1.0.0
```

### 2. API Key 설정

`.env` 파일 생성:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Gemini API Key 발급**:
- https://makersuite.google.com/app/apikey
- 무료 티어 사용 가능

### 3. 데이터 준비

크롤링한 WMS 문서를 `data/crawled/` 디렉토리에 배치:
```
data/crawled/
├── wms_manual_입고.txt
├── wms_manual_출고.txt
├── wms_faq.csv
└── wms_processes.json
```

### 4. 실행

```bash
python example.py
```

**출력**:
- `data/output/wms_instruction_dataset.json` - Instruction Tuning 데이터셋
- `data/vectordb/wms_vectordb/` - FAISS 벡터 DB (재사용 가능)

---

## 📊 생성되는 데이터 형식

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "재고 실사할 때 시간이 너무 오래 걸리는데, 이걸 줄일 수 있나요?"
      },
      {
        "role": "assistant",
        "content": "WMS를 활용하면 재고 실사 시간을 크게 단축할 수 있습니다. 바코드 스캔을 통해 실시간으로 재고를 확인하고..."
      }
    ],
    "metadata": {
      "topic": "재고 정확도 향상 방법",
      "persona": "김영수 (중소 물류센터 관리자)",
      "persona_background": "전통적인 수기 관리에서 WMS 도입 검토 중",
      "context_used": "재고 실사는...(참고 문서 일부)",
      "num_docs_retrieved": 3,
      "created_at": "2025-10-02T..."
    }
  }
]
```

---

## 🔧 커스터마이징

### 페르소나 추가

`example.py`의 `LogisticsPersona.PERSONAS`에 추가:

```python
{
    "name": "새로운 페르소나",
    "background": "배경 설명",
    "concerns": ["관심사1", "관심사2"],
    "question_style": "질문 스타일"
}
```

### 주제 수정

`main()` 함수의 `TOPICS` 리스트 수정:

```python
TOPICS = [
    "커스텀 주제 1",
    "커스텀 주제 2",
    # ...
]
```

### 생성 수량 조절

```python
dataset = builder.build_dataset(
    topics=TOPICS,
    questions_per_topic=3  # 페르소나당 질문 수 조절
)
```

---

## 🎓 파인튜닝 진행

생성된 데이터셋으로 SOLAR 모델 파인튜닝:

### 1. 데이터셋 준비

```python
# training/solar_finetuning.py 에서

def load_wms_dataset():
    with open("data/output/wms_instruction_dataset.json", "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)
```

### 2. 기존 체크포인트에서 재개

```python
config = KTCloudH100Config()
config.resume_from_checkpoint = "MyeongHo0621/solar-korean-wms/checkpoint-1000"
```

### 3. 학습 파라미터 조정

```python
config.num_train_epochs = 2  # 도메인 특화는 적은 에포크
config.learning_rate = 1e-5  # 낮은 학습률로 안정성 확보
```

### 4. 실행

```bash
python training/solar_finetuning.py
```

---

## 📈 예상 결과

### 데이터셋 규모

- **주제**: 10개
- **페르소나**: 5명
- **질문/주제**: 페르소나당 2-3개
- **총 QA 수**: 약 100-150개

생성 시간: 약 30-60분 (Gemini API 속도에 따라 변동)

### 파인튜닝 후 성능 개선

**Before** (일반 한국어 모델):
```
Q: 크로스도킹이 뭔가요?
A: 크로스도킹은 물류 용어입니다... (일반적인 설명)
```

**After** (WMS 특화 모델):
```
Q: 크로스도킹이 뭔가요?
A: 크로스도킹은 입고된 상품을 창고에 보관하지 않고 즉시 출고 구역으로 이동시키는 
   물류 전략입니다. 보관 비용과 처리 시간을 줄일 수 있어 신선식품이나 
   빠른 회전율이 필요한 상품에 효과적입니다. 다만 정확한 입출고 스케줄링과 
   실시간 재고 가시성이 필수적입니다...
```

---

## 🔄 전체 워크플로우

### Phase 1: 데이터 수집 및 준비 (1-2일)
- [ ] WMS 문서 크롤링
- [ ] 문서 정제 및 포맷 통일
- [ ] `data/crawled/` 디렉토리에 배치

### Phase 2: QA 데이터셋 생성 (반나절)
- [ ] Gemini API Key 발급
- [ ] `example.py` 실행
- [ ] 생성된 데이터셋 품질 검수

### Phase 3: 데이터 보강 (선택, 1-2일)
- [ ] 실제 고객 문의 추가
- [ ] WMS 전문가 검수 및 수정
- [ ] 추가 시나리오 반영

### Phase 4: 파인튜닝 (4-6시간)
- [ ] 기존 체크포인트 로드
- [ ] WMS 데이터셋으로 재학습
- [ ] 검증 및 최적 체크포인트 선택

### Phase 5: 배포 준비 (1일)
- [ ] 모델 병합 (LoRA → Full model)
- [ ] 양자화 (FP8)
- [ ] vLLM/SGLang 서버 설정

### Phase 6: 프로덕션 배포 (1-2일)
- [ ] AWS GPU 인스턴스 설정
- [ ] FastAPI 서버 구축
- [ ] 모니터링 및 로깅 설정

---

## 💡 Best Practices

### 1. 데이터 품질 관리
- ✅ 생성된 QA는 반드시 샘플 검수
- ✅ 명백히 틀린 답변은 제거 또는 수정
- ✅ 실제 고객 문의를 추가하여 현실성 향상

### 2. 페르소나 활용
- ✅ 실제 타겟 고객에 맞춰 페르소나 조정
- ✅ 질문 스타일이 너무 유사하면 다양성 추가
- ✅ 난이도를 다양하게 (초보자 ~ 전문가)

### 3. VectorDB 관리
- ✅ 문서 업데이트 시 FAISS 재구축
- ✅ 청크 크기(chunk_size)는 500자 권장
- ✅ 검색 결과가 부적절하면 임베딩 모델 변경 고려

### 4. 파인튜닝 전략
- ✅ 기존 일반 지식 손실 방지: 낮은 학습률 사용
- ✅ 과적합 방지: 1-2 에포크로 제한
- ✅ 검증 손실 모니터링: 증가하면 즉시 중단

---

## 🐛 트러블슈팅

### Q1: Gemini API 에러 (429 Too Many Requests)
**해결**: 요청 간 딜레이 추가
```python
import time
time.sleep(2)  # 각 요청 사이 2초 대기
```

### Q2: FAISS 검색 결과가 부정확함
**해결**: 임베딩 모델 변경 또는 청크 크기 조정
```python
# 다른 한국어 임베딩 모델 시도
EMBEDDING_MODEL = "BM-K/KoSimCSE-roberta"
```

### Q3: SOLAR 답변이 너무 짧거나 길음
**해결**: `max_tokens` 파라미터 조정
```python
answer_result = self.a_gen.generate_answer(
    question,
    max_tokens=500  # 조절
)
```

### Q4: GPU 메모리 부족
**해결**: 모델 로딩 시 양자화 적용
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

---

## 📚 참고 자료

### LLM Fine-tuning
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Langchain & RAG
- [Langchain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)

### Gemini API
- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Guide](https://ai.google.dev/docs)

### SOLAR Model
- [SOLAR-10.7B on Hugging Face](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)

---

## 🤝 기여 및 문의

이 파이프라인에 대한 질문이나 개선 제안이 있으시면:
- Issue를 등록하거나
- Pull Request를 보내주세요

---

## 📝 라이선스

- SOLAR Model: Apache 2.0
- Gemini API: Google Terms of Service
- 이 프로젝트 코드: MIT License

---

## 🎉 다음 단계

1. ✅ QA 데이터셋 생성 완료
2. ⏭️ **WMS 특화 파인튜닝 진행**
3. ⏭️ 모델 병합 및 양자화
4. ⏭️ vLLM/SGLang 배포
5. ⏭️ FastAPI 서버 구축
6. ⏭️ 프로덕션 모니터링

**현재 위치**: Phase 2 완료 → Phase 4 진행 예정

---

**Made with ❤️ for WMS Domain Specialization**
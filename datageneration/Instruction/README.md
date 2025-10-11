# WMS Instruction Tuning Dataset Generator

WMS 도메인 특화 LLM 파인튜닝을 위한 고품질 QA 데이터셋 자동 생성 도구

## 프로젝트 구조

```
Instruction/
├── README.md                    # 이 파일
├── requirements.txt             # 의존성 패키지
├── .env.example                 # 환경변수 예시
│
├── config.py                    # 설정 파일
├── personas.py                  # 페르소나 및 주제 정의
│
├── vectordb_builder.py          # FAISS 벡터 DB 구축
├── question_generator.py        # Gemini 질문 생성
├── answer_generator.py          # SOLAR 답변 생성
├── dataset_builder.py           # 데이터셋 빌더
├── main.py                      # 메인 실행 파일
│
└── data/
    ├── crawled/                 # 크롤링한 원본 데이터
    ├── vectordb/                # FAISS 벡터 DB
    └── output/                  # 생성된 데이터셋
```

## 역할 분담

- **Gemini 1.5 Flash**: 물류 현업자 역할로 질문 생성 (VectorDB 없음)
- **SOLAR + FAISS**: WMS 전문가 역할로 답변 생성 (VectorDB 검색 기반)

## Quick Start

### 1. 환경 설정

```bash
# 패키지 설치 (필수 패키지만)
pip install faiss-cpu==1.8.0 python-dotenv

# 환경변수 설정
# .env 파일에 GEMINI_API_KEY 입력
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 2. FAISS 벡터 DB 준비

프로젝트 루트에 `faiss_storage` 디렉토리가 있어야 합니다:

```
tesseract/
├── faiss_storage/
│   ├── config.json
│   ├── documents.json
│   ├── metadata.json
│   └── warehouse_automation_knowledge.index
└── Instruction/
```

### 3. 테스트

```bash
# FAISS 로딩 테스트
python vectordb_loader.py

# 전체 파이프라인 실행
python main.py
```

출력: `data/output/wms_instruction_dataset.json`

## 생성 프로세스

1. **벡터 DB 구축**: 크롤링 데이터 → FAISS 벡터 DB
2. **질문 생성**: Gemini로 5가지 페르소나 기반 질문 생성
3. **답변 생성**: SOLAR가 VectorDB 검색 후 답변 생성
4. **데이터셋 저장**: Instruction Tuning 형식으로 저장

## 개별 모듈 테스트

### 벡터 DB 구축만 실행

```bash
python vectordb_builder.py
```

### 질문 생성만 테스트

```bash
python question_generator.py
```

### 답변 생성만 테스트

```bash
python answer_generator.py
```

## 설정 커스터마이징

`config.py`에서 다음 설정을 변경할 수 있습니다:

### 데이터셋 생성 설정
- `QUESTIONS_PER_TOPIC`: 페르소나당 질문 수 (기본: 2)
- `MAX_ANSWER_TOKENS`: 최대 답변 길이 (기본: 600)

### SOLAR 생성 파라미터 (답변 품질 조정)
- `SOLAR_TEMPERATURE`: 0.5 (낮을수록 일관성↑, 높을수록 창의성↑)
- `SOLAR_TOP_P`: 0.85
- `SOLAR_TOP_K`: 30
- `SOLAR_REPETITION_PENALTY`: 1.15

### FAISS 설정
- `CHUNK_SIZE`: 문서 청크 크기 (기본: 500)
- `EMBEDDING_MODEL`: 임베딩 모델 (기본: jhgan/ko-sroberta-multitask)

## 페르소나 수정

`personas.py`에서 페르소나와 주제를 커스터마이징할 수 있습니다.

## 데이터 형식

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "재고 실사할 때 시간이 너무 오래 걸리는데..."
      },
      {
        "role": "assistant",
        "content": "WMS를 활용하면 재고 실사 시간을..."
      }
    ],
    "metadata": {
      "topic": "재고 정확도 향상 방법",
      "persona": "김영수 (중소 물류센터 관리자)",
      "num_docs_retrieved": 3
    }
  }
]
```

## 다음 단계

생성된 데이터셋으로 SOLAR 모델 파인튜닝을 진행하세요.

## 라이선스

MIT License


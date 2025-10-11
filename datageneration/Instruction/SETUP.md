# 설정 가이드

## 1. 기존 FAISS 벡터 DB 복사

기존에 있는 `faiss_storage` 디렉토리를 복사해야 합니다:

```bash
# 기존 FAISS 디렉토리를 Instruction/data/vectordb/로 복사
cp -r /path/to/your/faiss_storage Instruction/data/vectordb/

# 또는 심볼릭 링크 사용
ln -s /path/to/your/faiss_storage Instruction/data/vectordb/faiss_storage
```

복사 후 다음과 같은 구조가 되어야 합니다:

```
Instruction/data/vectordb/faiss_storage/
├── config.json
├── documents.json
├── metadata.json
└── warehouse_automation_knowledge.index
```

## 2. FAISS 구조 확인

`config.json` 예시:
```json
{
  "dimension": 768,
  "total_documents": 14383,
  "embedding_model": "jhgan/ko-sroberta-multitask",
  "index_type": "HNSW",
  "created_at": "2025-10-01T19:55:09.462679"
}
```

## 3. 테스트

벡터 DB가 제대로 로딩되는지 테스트:

```bash
cd Instruction
python vectordb_loader.py
```

정상 출력 예시:
```
기존 FAISS 벡터 DB 로딩...
  총 문서 수: 14383
  임베딩 모델: jhgan/ko-sroberta-multitask
  차원: 768
  인덱스 타입: HNSW
  벡터 수: 14383
✓ FAISS 벡터 DB 로딩 완료
```

## 4. 전체 파이프라인 실행

```bash
python main.py
```

## 주의사항

- 기존 FAISS는 `jhgan/ko-sroberta-multitask` 임베딩 모델을 사용합니다
- 문서가 로봇 관련 논문인 경우, WMS 질문에 대한 답변 품질이 낮을 수 있습니다
- WMS 관련 문서로 새로운 FAISS를 구축하려면 `vectordb_builder.py`를 사용하세요



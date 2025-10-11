"""기존 FAISS 벡터 DB 로더"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

from transformers import AutoModel, AutoTokenizer
from config import config

# FAISS import (GPU 우선, CPU fallback)
FAISS_GPU_AVAILABLE = False
try:
    import faiss
    # GPU 사용 가능 여부 체크
    if hasattr(faiss, 'StandardGpuResources'):
        try:
            res = faiss.StandardGpuResources()
            FAISS_GPU_AVAILABLE = True
            print("✓ faiss-gpu 사용 가능")
        except Exception as e:
            print(f"✓ faiss 로드 (GPU 초기화 실패, CPU 모드): {str(e)[:50]}...")
            FAISS_GPU_AVAILABLE = False
    else:
        print("✓ faiss-cpu 사용")
except ImportError:
    raise ImportError(
        "FAISS 패키지가 설치되지 않았습니다!\n"
        "GPU: pip install faiss-gpu-cu12\n"
        "CPU: pip install faiss-cpu"
    )


class FAISSVectorDBLoader:
    """기존 FAISS 벡터 DB 로딩 및 검색"""
    
    def __init__(self):
        print("\n기존 FAISS 벡터 DB 로딩...")
        
        # 경로 확인
        self.faiss_dir = config.FAISS_DIR
        if not self.faiss_dir.exists():
            raise FileNotFoundError(
                f"FAISS 디렉토리가 없습니다: {self.faiss_dir}\n"
                f"faiss_storage 디렉토리를 {config.VECTORDB_DIR}에 복사하세요."
            )
        
        # Config 로드
        config_path = self.faiss_dir / config.FAISS_CONFIG_FILE
        with open(config_path, 'r', encoding='utf-8') as f:
            self.db_config = json.load(f)
        
        print(f"  총 문서 수: {self.db_config['total_documents']}")
        print(f"  임베딩 모델: {self.db_config['embedding_model']}")
        print(f"  차원: {self.db_config['dimension']}")
        
        # Documents 로드
        documents_path = self.faiss_dir / config.FAISS_DOCUMENTS_FILE
        with open(documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        # Metadata 로드
        metadata_path = self.faiss_dir / config.FAISS_METADATA_FILE
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # FAISS 인덱스 로드
        index_path = str(self.faiss_dir / config.FAISS_INDEX_FILE)
        self.index = faiss.read_index(index_path)
        
        # GPU 사용 가능하면 GPU로 이동
        if FAISS_GPU_AVAILABLE and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print(f"  인덱스 타입: {self.db_config['index_type']} (GPU)")
            except Exception as e:
                print(f"  GPU 로딩 실패, CPU 사용: {e}")
                print(f"  인덱스 타입: {self.db_config['index_type']} (CPU)")
        else:
            print(f"  인덱스 타입: {self.db_config['index_type']} (CPU)")
        
        print(f"  벡터 수: {self.index.ntotal}")
        
        # 임베딩 모델 로드
        print(f"\n임베딩 모델 로딩: {self.db_config['embedding_model']}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.db_config['embedding_model'])
        self.embedding_model = AutoModel.from_pretrained(
            self.db_config['embedding_model'],
            use_safetensors=True,
            trust_remote_code=True
        ).to(self.device)
        
        if self.device == 'cuda':
            self.embedding_model = self.embedding_model.half()
        
        self.embedding_model.eval()
        
        print(f"  디바이스: {self.device}")
        print("✓ FAISS 벡터 DB 로딩 완료")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        # 토크나이즈
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 임베딩 생성
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # 정규화
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def search_relevant_docs(self, query: str, k: int = 3) -> List[str]:
        """질문과 관련된 문서 검색"""
        
        # 쿼리 임베딩
        query_embedding = self._encode_text(query)
        
        # FAISS 검색
        distances, indices = self.index.search(query_embedding, k)
        
        # 문서 반환
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def search_with_scores(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """질문과 관련된 문서 + 유사도 점수"""
        
        # 쿼리 임베딩
        query_embedding = self._encode_text(query)
        
        # FAISS 검색
        distances, indices = self.index.search(query_embedding, k)
        
        # 문서 + 점수 반환
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                # 거리를 유사도로 변환 (코사인 유사도)
                similarity = 1 - dist
                results.append((self.documents[idx], float(similarity)))
        
        return results
    
    def get_metadata(self, doc_index: int) -> dict:
        """문서 메타데이터 가져오기"""
        if doc_index < len(self.metadata):
            return self.metadata[doc_index]
        return {}
    
    def get_stats(self) -> dict:
        """벡터 DB 통계"""
        return {
            "total_documents": self.db_config['total_documents'],
            "embedding_model": self.db_config['embedding_model'],
            "dimension": self.db_config['dimension'],
            "index_type": self.db_config['index_type'],
            "index_vectors": self.index.ntotal
        }


if __name__ == "__main__":
    # 테스트
    try:
        loader = FAISSVectorDBLoader()
        
        # 통계 출력
        stats = loader.get_stats()
        print("\n벡터 DB 통계:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 검색 테스트
        test_query = "WMS 도입 비용"
        print(f"\n검색 테스트: {test_query}")
        
        results = loader.search_with_scores(test_query, k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[{i}] 유사도: {score:.4f}")
            print(f"문서: {doc[:200]}...")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n다음 단계:")
        print("1. 기존 faiss_storage 디렉토리를 Instruction/data/vectordb/로 복사")
        print("2. 다시 실행")



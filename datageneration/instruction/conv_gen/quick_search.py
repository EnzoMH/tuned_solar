"""
Quick Search Optimizer for FAISS
FAISS 검색 품질 및 속도 최적화 유틸리티 (GPU 지원)
"""

from typing import List, Dict, Optional
import re
import torch
import faiss
import numpy as np


class SearchOptimizer:
    """FAISS 검색 최적화 헬퍼 클래스"""
    
    # 제외할 기술 용어들 (너무 저수준 기술 문서 필터링)
    TECHNICAL_FILTERS = [
        'HARQ', 'packet', 'protocol', 'latency ms', 'throughput',
        'algorithm complexity', 'big-O', 'neural network architecture',
        'tensor', 'backpropagation', 'gradient descent',
        'UE', 'base station', '5G communication', 'RIS',
        'ADMM', 'GADMM', 'quantization', 'GPTQ'
    ]
    
    # WMS 관련 키워드
    WMS_KEYWORDS = [
        'WMS', '창고관리', '물류', '재고', '입출고', '피킹',
        '창고', '배송', '주문', '출하', '입고', '창고자동화',
        '물류센터', '유통', '배송센터'
    ]
    
    def __init__(self, distance_threshold: float = 1.4):
        """
        Args:
            distance_threshold: FAISS 검색 결과의 최대 허용 거리 (낮을수록 엄격)
        """
        self.distance_threshold = distance_threshold
        self.cache = {}
        self.cache_size = 100
    
    def enhance_query(self, question: str) -> str:
        """
        검색 쿼리를 WMS 맥락으로 확장
        
        Args:
            question: 원본 질문
            
        Returns:
            확장된 질문
        """
        # 이미 WMS 키워드가 충분히 있으면 그대로 사용
        wms_count = sum(1 for kw in self.WMS_KEYWORDS if kw.lower() in question.lower())
        
        if wms_count >= 2:
            return question
        
        # WMS 맥락 추가
        enhanced = f"WMS 창고관리시스템 물류 분야에서 {question}"
        return enhanced
    
    def filter_irrelevant_contexts(
        self, 
        contexts: List[Dict],
        question: str
    ) -> List[Dict]:
        """
        관련성 낮은 컨텍스트 필터링
        
        Args:
            contexts: FAISS 검색 결과
            question: 원본 질문
            
        Returns:
            필터링된 컨텍스트
        """
        filtered = []
        
        for ctx in contexts:
            content = ctx['content']
            distance = ctx.get('distance', 0)
            
            # 1. Distance threshold 체크
            if distance > self.distance_threshold:
                continue
            
            # 2. 기술 용어 필터링 (너무 많으면 제외)
            tech_count = sum(1 for term in self.TECHNICAL_FILTERS 
                            if term.lower() in content.lower())
            if tech_count >= 3:  # 3개 이상 기술 용어 → 제외
                continue
            
            # 3. WMS 관련성 체크
            wms_count = sum(1 for kw in self.WMS_KEYWORDS 
                           if kw.lower() in content.lower())
            if wms_count == 0 and distance > 1.2:  # WMS 키워드 없고 거리도 멀면 제외
                continue
            
            filtered.append(ctx)
        
        return filtered
    
    def rerank_by_relevance(
        self,
        contexts: List[Dict],
        question: str
    ) -> List[Dict]:
        """
        질문 키워드 기반 재랭킹
        
        Args:
            contexts: 필터링된 컨텍스트
            question: 원본 질문
            
        Returns:
            재랭킹된 컨텍스트
        """
        # 질문에서 핵심 키워드 추출
        question_lower = question.lower()
        question_keywords = set(re.findall(r'\w+', question_lower))
        
        # 각 컨텍스트의 관련성 점수 계산
        scored_contexts = []
        for ctx in contexts:
            content_lower = ctx['content'].lower()
            
            # 키워드 매칭 점수
            keyword_score = sum(1 for kw in question_keywords 
                               if len(kw) > 2 and kw in content_lower)
            
            # WMS 관련성 점수
            wms_score = sum(1 for kw in self.WMS_KEYWORDS 
                           if kw.lower() in content_lower)
            
            # Distance 점수 (낮을수록 좋음)
            distance_score = 2.0 - ctx.get('distance', 1.0)
            
            # 총점
            total_score = keyword_score * 2 + wms_score + distance_score
            
            scored_contexts.append((total_score, ctx))
        
        # 점수 순 정렬
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        
        return [ctx for score, ctx in scored_contexts]
    
    def optimize_search(
        self,
        question: str,
        raw_contexts: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        전체 검색 최적화 파이프라인
        
        Args:
            question: 원본 질문
            raw_contexts: FAISS 원시 검색 결과
            top_k: 최종 반환할 컨텍스트 수
            
        Returns:
            최적화된 컨텍스트
        """
        # 캐시 확인
        cache_key = f"{question}_{len(raw_contexts)}_{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 1. 관련성 낮은 것 필터링
        filtered = self.filter_irrelevant_contexts(raw_contexts, question)
        
        # 2. 재랭킹
        reranked = self.rerank_by_relevance(filtered, question)
        
        # 3. Top-K 선택
        result = reranked[:top_k]
        
        # 캐시 저장
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()


# GPU 유틸리티 함수들
class GPUOptimizer:
    """GPU FAISS 최적화 헬퍼"""
    
    @staticmethod
    def check_gpu_available() -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            return torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
        except:
            return False
    
    @staticmethod
    def move_index_to_gpu(
        cpu_index: faiss.Index, 
        gpu_id: int = 0
    ) -> Optional[faiss.Index]:
        """
        FAISS 인덱스를 GPU로 이동
        
        Args:
            cpu_index: CPU FAISS 인덱스
            gpu_id: 사용할 GPU ID (기본값: 0)
            
        Returns:
            GPU 인덱스 또는 실패 시 None
        """
        if not GPUOptimizer.check_gpu_available():
            print("⚠️ GPU not available, using CPU index")
            return None
        
        try:
            # GPU 리소스 생성
            res = faiss.StandardGpuResources()
            
            # 임시 메모리 설정 (성능 향상)
            res.setTempMemory(256 * 1024 * 1024)  # 256MB
            
            # CPU -> GPU 변환
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
            
            print(f"✓ FAISS index moved to GPU:{gpu_id}")
            return gpu_index
        
        except Exception as e:
            print(f"⚠️ Failed to move index to GPU: {e}")
            return None
    
    @staticmethod
    def get_gpu_info():
        """GPU 정보 출력"""
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return
        
        print(f"\n{'='*60}")
        print("GPU Information:")
        print(f"{'='*60}")
        print(f"Device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            
            if torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info(i)
                print(f"  Free memory: {free / 1024**3:.2f} GB")
                print(f"  Used memory: {(total - free) / 1024**3:.2f} GB")
        
        print(f"{'='*60}\n")


# 전역 인스턴스 (싱글톤 패턴)
_optimizer_instance = None

def get_search_optimizer(distance_threshold: float = 1.4) -> SearchOptimizer:
    """SearchOptimizer 싱글톤 인스턴스 반환"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = SearchOptimizer(distance_threshold)
    return _optimizer_instance


"""
Answer Maker v2 - EXAONE 모델 기반 답변 생성기
RAG(Retrieval-Augmented Generation)를 활용한 AI Assistant 스타일 답변
"""

import torch
import json
import faiss
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from quick_search import get_search_optimizer, GPUOptimizer
from hw_optimization import GPUTimer, GPUMonitor


class AnswerMakerV2:
    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        faiss_path: str = "/home/work/tesseract/faiss_storage",
        embedding_model: str = "jhgan/ko-sroberta-multitask"
    ):
        """EXAONE 4.0 모델 기반 답변 생성기 초기화"""
        print(f"\n{'='*80}", flush=True)
        print(f"Answer Maker v2 초기화 (Model: {model_name})", flush=True)
        print(f"Mode: AI Assistant + FAISS RAG", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # EXAONE 4.0 모델 로딩
        print(f"Loading EXAONE 4.0 model...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map="auto"
        )
        print(f"✓ Model loaded: {self.model.device}", flush=True)
        print(f"✓ Memory: {self.model.get_memory_footprint() / 1024**3:.2f} GB\n", flush=True)
        
        # FAISS 벡터 DB 로딩 (GPU 최적화)
        print(f"Loading FAISS vector store...", flush=True)
        
        # GPU 정보 출력
        if torch.cuda.is_available():
            GPUOptimizer.get_gpu_info()
        
        # Embedding 모델 로딩 (GPU 우선)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=device
        )
        print(f"✓ Embedding model on: {device}", flush=True)
        
        # FAISS 인덱스 직접 로딩
        faiss_dir = Path(faiss_path)
        index_file = faiss_dir / "warehouse_automation_knowledge.index"
        documents_file = faiss_dir / "documents.json"
        metadata_file = faiss_dir / "metadata.json"
        
        # CPU 인덱스 로드
        cpu_index = faiss.read_index(str(index_file))
        
        # GPU로 이동 시도
        gpu_index = GPUOptimizer.move_index_to_gpu(cpu_index, gpu_id=0)
        self.faiss_index = gpu_index if gpu_index is not None else cpu_index
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"✓ FAISS loaded: {len(self.documents)} documents\n", flush=True)
        
        # 검색 최적화 도구 초기화
        self.search_optimizer = get_search_optimizer(distance_threshold=1.4)
        
        # GPU 타이머 초기화 (정확한 시간 측정)
        self.gpu_timer = GPUTimer(device=str(self.model.device))
        
        self.model_name = model_name
    
    def retrieve_context(self, question: str, k: int = 5) -> List[Dict]:
        """질문과 관련된 컨텍스트를 FAISS에서 검색 (최적화 적용)"""
        # 1. 질문 확장 (WMS 맥락 추가)
        enhanced_question = self.search_optimizer.enhance_query(question)
        
        # 2. 임베딩 생성
        query_vector = self.embedding_model.encode([enhanced_question], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype('float32')
        
        # 3. FAISS 검색 (더 많이 검색해서 필터링)
        search_k = k * 2  # 10개 검색 후 필터링
        distances, indices = self.faiss_index.search(query_vector, search_k)
        
        # 4. 원시 결과 구조화
        raw_contexts = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                raw_contexts.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'rank': i + 1,
                    'distance': float(distances[0][i])
                })
        
        # 5. 검색 최적화 (필터링 + 재랭킹)
        optimized_contexts = self.search_optimizer.optimize_search(
            question=question,
            raw_contexts=raw_contexts,
            top_k=k
        )
        
        # 6. Rank 재정렬
        for i, ctx in enumerate(optimized_contexts, 1):
            ctx['rank'] = i
        
        return optimized_contexts
    
    def create_answer_prompt(self, question: str, contexts: List[Dict]) -> str:
        """RAG 기반 답변 생성용 프롬프트 (참고자료 언급 금지)"""
        # 컨텍스트 결합 (번호 제거)
        context_text = "\n\n".join([
            f"{ctx['content'][:500]}"
            for ctx in contexts
        ])
        
        prompt = f"""질문: {question}

배경 정보:
{context_text}

위 정보를 바탕으로 답변하되, 다음 규칙을 반드시 준수하세요:

 ## 절대 금지사항: ##
- "참고자료", "자료", "문서", "참고" 등의 단어 사용 금지
- 인용 표시나 출처 언급 금지  
- "~에 따르면", "~에서 언급된", "~에 의하면" 등의 표현 금지
- [참고자료 1], [사례 1] 같은 번호 표시 금지

## 답변 방식: ##
- 자연스럽게 정보를 녹여서 설명 (마치 본인의 지식인 것처럼)
- 구체적인 숫자와 사례는 직접 설명
- 전문적이면서 대화체 톤 유지
- 장단점을 균형있게 제시
- 반드시 한국어로만 작성

예시:
❌ 나쁜 답변: "참고자료 1에 따르면 WMS 비용은..."
✅ 좋은 답변: "WMS 시스템 도입 비용은 창고 면적당 약 15-20만원 정도입니다..."

한국어 답변:"""
        
        return prompt
    
    def generate_answer(
        self,
        question: str,
        num_contexts: int = 5,
        max_new_tokens: int = 1500
    ) -> Dict:
        """질문에 대한 답변 생성 (FAISS RAG 사용)"""
        print(f"\n{'='*80}", flush=True)
        print(f"Q: {question}", flush=True)
        print(f"{'='*80}", flush=True)
        
        # 1. FAISS 컨텍스트 검색
        contexts = self.retrieve_context(question, k=num_contexts)
        
        # 2. RAG 프롬프트 생성
        prompt = self.create_answer_prompt(question, contexts)
        
        # EXAONE messages 형식 적용
        messages = [
            {
                "role": "system", 
                "content": """당신은 WMS 전문가입니다.
답변할 때 절대로 '참고자료', '자료', '문서', '참고' 등을 언급하지 마세요.
모든 정보를 당신의 지식인 것처럼 자연스럽게 설명하세요.
반드시 한국어로만 답변하세요."""
            },
            {"role": "user", "content": prompt}
        ]
        
        # 토크나이저 적용
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # 생성 (GPU 동기화 포함 정확한 시간 측정)
        with self.gpu_timer.measure() as timer:
            outputs = self.model.generate(
                input_ids.to(self.model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        inference_time = timer["elapsed"]
        
        # 디코딩
        answer = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"A: {answer[:1500]}{'...' if len(answer) > 200 else ''}", flush=True)
        print(f"⏱ {inference_time:.1f}s\n", flush=True)
        
        # 결과 반환
        result = {
            'question': question,
            'answer': answer,
            'num_contexts': len(contexts),
            'contexts': contexts,
            'inference_time_sec': inference_time,
            'input_tokens': input_ids.shape[1],
            'output_tokens': outputs.shape[1] - input_ids.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def generate_answers_batch(
        self,
        questions: List[str],
        num_contexts: int = 5
    ) -> List[Dict]:
        """여러 질문에 대한 답변 일괄 생성"""
        print(f"\n{'#'*80}", flush=True)
        print(f"Batch Answer Generation (EXAONE + FAISS RAG)", flush=True)
        print(f"Total questions: {len(questions)}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---", flush=True)
            
            result = self.generate_answer(
                question=question,
                num_contexts=num_contexts
            )
            
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """결과 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_file}", flush=True)
    
    def cleanup(self):
        """메모리 정리"""
        del self.model
        del self.tokenizer
        del self.embedding_model
        del self.faiss_index
        del self.documents
        del self.metadata
        self.search_optimizer.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """테스트 실행"""
    # Answer Maker v2 초기화
    am = AnswerMakerV2()
    
    # 테스트 질문들 (AI에게 묻는 스타일)
    test_questions = [
        "WMS 시스템 도입 시 초기 투자 비용은 대략 어느 정도인가요?",
        "재고 정확도를 개선하기 위한 효과적인 방법들을 알려주세요.",
        "중소기업도 WMS를 도입할 수 있나요? 비용 부담은 어떤가요?"
    ]
    
    # FAISS RAG 기반 답변 생성
    results = am.generate_answers_batch(test_questions)
    
    # 저장
    am.save_results(results, "generated_answers_v2.json")
    
    # 정리
    am.cleanup()


if __name__ == "__main__":
    main()


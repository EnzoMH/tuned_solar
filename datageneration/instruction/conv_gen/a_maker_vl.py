"""
Answer Maker vLLM - EXAONE 모델 기반 답변 생성기 (H100 최적화)
vLLM + Tensor Parallelism (2 GPUs) + FAISS RAG
"""

import torch
import json
import faiss
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from .quick_search import get_search_optimizer, GPUOptimizer
from .hw_optimization.hardware_optimization import GPUTimer, GPUMonitor


class AnswerMakerVLLM:
    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        faiss_path: str = "/home/work/tesseract/faiss_storage",
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.85
    ):
        """vLLM 기반 EXAONE 답변 생성기 초기화 (H100 2개 활용)"""
        print(f"\n{'='*80}", flush=True)
        print(f"Answer Maker vLLM 초기화 (Model: {model_name})", flush=True)
        print(f"Mode: vLLM + Tensor Parallelism ({tensor_parallel_size} GPUs) + FAISS RAG", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # GPU 정보 출력
        if torch.cuda.is_available():
            GPUMonitor.print_all_gpus()
        
        # vLLM으로 EXAONE 모델 로딩
        print(f"Loading EXAONE model with vLLM...", flush=True)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,  # H100 2개 활용
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",  # H100 최적
            max_model_len=4096,
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded on {tensor_parallel_size} GPU(s) with vLLM\n", flush=True)
        
        # Sampling 파라미터 설정 (한국어 최적화)
        self.sampling_params = SamplingParams(
            temperature=0.1,  # 한국어는 낮게 (공식 권장)
            top_p=0.95,
            max_tokens=800,  # 1500 → 800으로 조정
            repetition_penalty=1.15,
            skip_special_tokens=True
        )
        
        # FAISS 벡터 DB 로딩
        print(f"Loading FAISS vector store...", flush=True)
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=device
        )
        print(f"✓ Embedding model on: {device}", flush=True)
        
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
        
        # GPU 타이머 초기화
        self.gpu_timer = GPUTimer(device="cuda")
        
        self.model_name = model_name
    
    def retrieve_context(self, question: str, k: int = 5) -> List[Dict]:
        """질문과 관련된 컨텍스트를 FAISS에서 검색 (최적화 적용)"""
        # 1. 질문 확장 (WMS 맥락 추가)
        enhanced_question = self.search_optimizer.enhance_query(question)
        
        # 2. 임베딩 생성
        query_embedding = self.embedding_model.encode(
            enhanced_question,
            convert_to_numpy=True
        ).reshape(1, -1).astype('float32')
        
        # 3. FAISS 검색 (더 많이 검색해서 필터링)
        distances, indices = self.faiss_index.search(query_embedding, k * 2)
        
        # 4. 원시 컨텍스트 수집
        raw_contexts = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                raw_contexts.append({
                    'content': self.documents[idx],
                    'distance': float(distance),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                })
        
        # 5. 검색 최적화 적용 (필터링 + 리랭킹)
        optimized_contexts = self.search_optimizer.optimize_search(
            question=question,
            raw_contexts=raw_contexts,
            top_k=k
        )
        
        return optimized_contexts
    
    def create_answer_prompt(self, question: str, contexts: List[Dict]) -> str:
        """RAG 프롬프트 생성 (참고자료 언급 금지)"""
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
    
    def create_chat_messages(self, question: str, contexts: List[Dict]) -> List[Dict]:
        """EXAONE chat template용 메시지 생성"""
        prompt = self.create_answer_prompt(question, contexts)
        
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
        
        return messages
    
    def generate_answer(
        self,
        question: str,
        num_contexts: int = 5
    ) -> Dict:
        """단일 질문에 대한 답변 생성 (vLLM 사용)"""
        print(f"\n{'='*80}", flush=True)
        print(f"Q: {question}", flush=True)
        print(f"{'='*80}", flush=True)
        
        # 1. FAISS 컨텍스트 검색
        contexts = self.retrieve_context(question, k=num_contexts)
        
        # 2. Chat messages 생성
        messages = self.create_chat_messages(question, contexts)
        
        # 3. vLLM chat template 적용
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 4. vLLM 생성 (GPU 동기화 포함 정확한 시간 측정)
        with self.gpu_timer.measure() as timer:
            outputs = self.llm.generate([prompt], self.sampling_params)
        
        inference_time = timer["elapsed"]
        
        # 5. 결과 추출
        answer = outputs[0].outputs[0].text
        
        print(f"A: {answer[:1500]}{'...' if len(answer) > 1500 else ''}", flush=True)
        print(f"⏱ {inference_time:.1f}s\n", flush=True)
        
        # 6. 결과 반환
        result = {
            'question': question,
            'answer': answer,
            'num_contexts': len(contexts),
            'contexts': contexts,
            'inference_time_sec': inference_time,
            'output_tokens': len(tokenizer.encode(answer)),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def generate_answers_batch(
        self,
        questions: List[str],
        num_contexts: int = 5,
        batch_size: int = 8
    ) -> List[Dict]:
        """
        배치로 답변 생성 (vLLM continuous batching 활용)
        
        Args:
            questions: 질문 리스트
            num_contexts: 각 질문당 검색할 컨텍스트 수
            batch_size: vLLM 배치 크기
        
        Returns:
            답변 딕셔너리 리스트
        """
        print(f"\n{'#'*80}", flush=True)
        print(f"Batch Answer Generation (vLLM + FAISS RAG)", flush=True)
        print(f"Total questions: {len(questions)}", flush=True)
        print(f"Batch size: {batch_size}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        
        all_results = []
        
        # 배치 단위로 처리
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            
            print(f"--- Batch {i//batch_size + 1}/{(len(questions)-1)//batch_size + 1} ({len(batch_questions)} questions) ---\n", flush=True)
            
            # 1. 모든 질문에 대한 컨텍스트 검색
            batch_contexts = []
            for q in batch_questions:
                contexts = self.retrieve_context(q, k=num_contexts)
                batch_contexts.append(contexts)
            
            # 2. 모든 프롬프트 생성
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            batch_prompts = []
            for q, contexts in zip(batch_questions, batch_contexts):
                messages = self.create_chat_messages(q, contexts)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_prompts.append(prompt)
            
            # 3. vLLM 배치 생성 (매우 빠름!)
            with self.gpu_timer.measure() as timer:
                outputs = self.llm.generate(batch_prompts, self.sampling_params)
            
            batch_inference_time = timer["elapsed"]
            avg_time_per_question = batch_inference_time / len(batch_questions)
            
            # 4. 결과 수집
            for j, (q, contexts, output) in enumerate(zip(batch_questions, batch_contexts, outputs)):
                answer = output.outputs[0].text
                
                print(f"\n{'='*80}", flush=True)
                print(f"Q{i+j+1}: {q}", flush=True)
                print(f"{'='*80}", flush=True)
                print(f"A: {answer[:800]}{'...' if len(answer) > 800 else ''}", flush=True)
                print(f"⏱ {avg_time_per_question:.1f}s\n", flush=True)
                
                result = {
                    'question': q,
                    'answer': answer,
                    'num_contexts': len(contexts),
                    'contexts': contexts,
                    'inference_time_sec': avg_time_per_question,
                    'output_tokens': len(tokenizer.encode(answer)),
                    'timestamp': datetime.now().isoformat()
                }
                
                all_results.append(result)
            
            print(f"✓ Batch {i//batch_size + 1} completed in {batch_inference_time:.2f}s", flush=True)
            print(f"  Average: {avg_time_per_question:.2f}s per question\n", flush=True)
        
        print(f"✓ {len(all_results)}개 답변 생성 완료\n", flush=True)
        
        return all_results
    
    def cleanup(self):
        """메모리 정리"""
        self.search_optimizer.clear_cache()
        print("✓ Cleanup completed", flush=True)


# 편의 함수
def create_answer_maker_vllm(
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.85
):
    """vLLM 답변 생성기 팩토리 함수"""
    return AnswerMakerVLLM(
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )


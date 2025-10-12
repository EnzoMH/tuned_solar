"""
Answer Maker - EEVE 모델 기반 답변 생성기
RAG(Retrieval-Augmented Generation)를 활용한 고품질 답변 생성
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


class AnswerMaker:
    def __init__(
        self,
        model_name: str = "MyeongHo0621/eeve-vss-smh",
        faiss_path: str = "/home/work/tesseract/faiss_storage",
        embedding_model: str = "jhgan/ko-sroberta-multitask"
    ):
        """EEVE 모델 기반 답변 생성기 초기화"""
        print(f"\n{'='*80}", flush=True)
        print(f"Answer Maker 초기화 (Model: {model_name})", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # EEVE 모델 로딩
        print(f"Loading EEVE model...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print(f"✓ Model loaded: {self.model.device}", flush=True)
        print(f"✓ Memory: {self.model.get_memory_footprint() / 1024**3:.2f} GB\n", flush=True)
        
        # FAISS 벡터 DB 로딩 (직접 로드 방식)
        print(f"Loading FAISS vector store...", flush=True)
        
        # Embedding 모델 로딩
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # FAISS 인덱스 직접 로딩
        faiss_dir = Path(faiss_path)
        index_file = faiss_dir / "warehouse_automation_knowledge.index"
        documents_file = faiss_dir / "documents.json"
        metadata_file = faiss_dir / "metadata.json"
        
        self.faiss_index = faiss.read_index(str(index_file))
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"✓ FAISS loaded: {len(self.documents)} documents\n", flush=True)
        
        self.model_name = model_name
    
    def retrieve_context(self, question: str, k: int = 5) -> List[Dict]:
        """질문과 관련된 컨텍스트를 FAISS에서 검색 (직접 검색 방식)"""
        print(f"Retrieving context for question: {question[:60]}...", flush=True)
        
        # 질문 임베딩
        query_vector = self.embedding_model.encode([question], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype('float32')
        
        # FAISS 검색
        distances, indices = self.faiss_index.search(query_vector, k)
        
        # 결과 구조화
        contexts = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                contexts.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'rank': i + 1,
                    'distance': float(distances[0][i])
                })
        
        print(f"✓ Retrieved {len(contexts)} contexts\n", flush=True)
        return contexts
    
    def create_answer_prompt(self, question: str, contexts: List[Dict]) -> str:
        """RAG 기반 답변 생성용 프롬프트 작성 (실무자 경험 기반)"""
        # 컨텍스트 결합
        context_text = "\n\n".join([
            f"[사례 {ctx['rank']}]\n{ctx['content'][:500]}"
            for ctx in contexts
        ])
        
        prompt = f"""당신은 10년 이상 물류 시스템 구축 프로젝트를 진행해온 현장 전문가인 동시에 비즈니스마인드가 강점인 전문가입니다.
여러 고객사의 WMS 도입 프로젝트를 성공시킨 경험이 있습니다.

아래는 실제 프로젝트 사례와 현장 경험 자료입니다:
{context_text}

질문: {question}

위 사례들을 바탕으로 실무 경험을 공유하듯이 답변해주세요.

답변 작성 규칙:
1. "참고자료에 따르면" (X) → "실제 제가 진행한 프로젝트에서는", "경험상", "보통 ~합니다" (O)
2. 구체적인 숫자와 기간을 포함 (예: "평균 3-6개월", "약 30% 절감", "하루 500건 → 1,200건")
3. 실제 고객사 사례 느낌으로 (예: "A사의 경우", "중소 물류센터들은 보통")
4. "해야 합니다", "필수적입니다" 같은 교과서 표현 금지
5. 자연스러운 대화체로, 상담하듯이 설명
6. 장단점을 솔직하게 (좋은 점만 말하지 말고 주의할 점도 언급)
7. 실제 컨설턴트와 같이 정중하지만 자연스러운 대화체로 답변,

나쁜 답변 예시:
"참고자료에 따르면 WMS 시스템은 재고 관리 효율성을 향상시킵니다. 따라서 도입을 검토해야 합니다."

좋은 답변 예시:
"제가 작년에 진행한 중소 물류센터 프로젝트를 보면요, 재고 실사 시간이 하루 8시간에서 2시간으로 줄었습니다. 
다만 처음 3개월은 직원들이 적응하느라 오히려 더 느렸어요. 평균적으로 6개월 정도 지나면 확실한 효과가 보입니다."

답변:"""
        
        return prompt
    
    def generate_answer(
        self,
        question: str,
        num_contexts: int = 5,
        max_new_tokens: int = 512
    ) -> Dict:
        """질문에 대한 답변 생성 (항상 FAISS RAG 사용)"""
        print(f"\n{'='*80}", flush=True)
        print(f"Generating answer for: {question}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # 1. FAISS 컨텍스트 검색
        contexts = self.retrieve_context(question, k=num_contexts)
        
        # 2. RAG 프롬프트 생성
        prompt = self.create_answer_prompt(question, contexts)
        
        # EEVE 프롬프트 템플릿 적용
        full_prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {prompt}
Assistant: """
        
        # 토크나이징
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        # 생성
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.15,  # 최적 균형 (반복 감소 + 자연스러움 유지)
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        # 디코딩
        answer = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"Generated answer:\n{answer}\n", flush=True)
        print(f"Inference time: {inference_time:.2f} sec\n", flush=True)
        
        # 결과 반환
        result = {
            'question': question,
            'answer': answer,
            'num_contexts': len(contexts),
            'contexts': contexts,
            'inference_time_sec': inference_time,
            'input_tokens': inputs['input_ids'].shape[1],
            'output_tokens': outputs.shape[1] - inputs['input_ids'].shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def generate_answers_batch(
        self,
        questions: List[str],
        num_contexts: int = 5
    ) -> List[Dict]:
        """여러 질문에 대한 답변 일괄 생성 (항상 FAISS RAG 사용)"""
        print(f"\n{'#'*80}", flush=True)
        print(f"Batch Answer Generation (FAISS RAG)", flush=True)
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
    
    def process_question_file(
        self,
        input_file: str,
        output_file: str
    ):
        """질문 파일을 읽어 답변 생성 후 저장 (항상 FAISS RAG 사용)"""
        print(f"\n{'='*80}", flush=True)
        print(f"Processing question file: {input_file}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # 질문 파일 로딩
        with open(input_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # 질문만 추출
        if isinstance(questions_data, list):
            # 리스트 형태 처리
            if isinstance(questions_data[0], dict):
                questions = [q.get('question', q.get('q', '')) for q in questions_data]
            else:
                questions = questions_data
        else:
            raise ValueError("지원하지 않는 질문 파일 형식입니다.")
        
        print(f"Loaded {len(questions)} questions\n", flush=True)
        
        # 답변 생성 (항상 RAG)
        results = self.generate_answers_batch(questions)
        
        # Q-A 페어 형식으로 저장
        qa_pairs = []
        for result in results:
            qa_pairs.append({
                'question': result['question'],
                'answer': result['answer'],
                'metadata': {
                    'num_contexts': result['num_contexts'],
                    'inference_time_sec': result['inference_time_sec'],
                    'input_tokens': result['input_tokens'],
                    'output_tokens': result['output_tokens'],
                    'timestamp': result['timestamp']
                }
            })
        
        # 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}", flush=True)
        print(f"Q-A pairs saved to: {output_file}", flush=True)
        print(f"Total pairs: {len(qa_pairs)}", flush=True)
        print(f"{'='*80}\n", flush=True)
    
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """테스트 실행"""
    # Answer Maker 초기화
    am = AnswerMaker()
    
    # 테스트 질문들
    test_questions = [
        "창고 자동화 시스템에서 AGV의 역할은 무엇인가요?",
        "효율적인 재고 관리를 위한 방법에는 어떤 것들이 있나요?",
        "WMS(Warehouse Management System)의 주요 기능을 설명해주세요."
    ]
    
    # FAISS RAG 기반 답변 생성 (항상)
    results = am.generate_answers_batch(test_questions)
    
    # 저장
    am.save_results(results, "generated_answers.json")
    
    # 정리
    am.cleanup()


if __name__ == "__main__":
    main()


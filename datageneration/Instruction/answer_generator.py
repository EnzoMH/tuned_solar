"""SOLAR 기반 답변 생성기 (WMS 전문가 역할 + VectorDB)"""

import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import config


class SOLARAnswerGenerator:
    """SOLAR = WMS 전문가 (VectorDB 검색 후 답변)"""
    
    def __init__(self, vectordb_loader):
        self.vectordb = vectordb_loader
        
        print(f"\nSOLAR 모델 로딩 (WMS 전문가 역할)...")
        print(f"  Base: {config.SOLAR_BASE_MODEL}")
        print(f"  Adapter: {config.SOLAR_ADAPTER}")
        
        # 베이스 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.SOLAR_BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # LoRA 어댑터 로드
        self.model = PeftModel.from_pretrained(
            self.base_model,
            config.SOLAR_ADAPTER,
            torch_dtype=torch.bfloat16
        )
        
        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(config.SOLAR_ADAPTER)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ SOLAR 모델 로딩 완료")
    
    def search_relevant_docs(self, question: str, k: int = 3) -> List[str]:
        """질문 관련 문서 검색"""
        return self.vectordb.search_relevant_docs(question, k=k)
    
    def generate_answer(
        self,
        question: str,
        persona_name: str = None,
        persona_background: str = None,
        max_tokens: int = None
    ) -> Dict:
        """
        RAG 방식으로 답변 생성
        1. FAISS에서 관련 문서 검색
        2. Persona 정보를 고려하여 답변 생성
        """
        
        if max_tokens is None:
            max_tokens = config.MAX_ANSWER_TOKENS
        
        # Step 1: FAISS 검색
        relevant_docs = self.search_relevant_docs(question, k=3)
        
        # Persona 정보 구성
        persona_info = ""
        if persona_name and persona_background:
            persona_info = f"""
[질문자 정보]
- 이름: {persona_name}
- 배경: {persona_background}
"""
        
        if not relevant_docs:
            # 문서 없으면 일반 답변
            prompt = f"""당신은 WMS(창고관리시스템) 전문 컨설턴트입니다.
물류 현업자의 실무 질문에 구체적이고 실용적으로 답변하세요.
{persona_info}
질문: {question}

답변:"""
            context_used = None
        else:
            # 문서 기반 답변
            context = "\n\n".join(relevant_docs[:2])
            
            prompt = f"""당신은 WMS(창고관리시스템) 전문 컨설턴트입니다.
물류 현업자의 실무 질문에 구체적이고 실용적으로 답변하세요.

다음 참고 자료를 활용하되, 질문자의 상황에 맞게 답변하세요.
{persona_info}
[참고 자료]
{context[:1500]}

질문: {question}

답변:"""
            context_used = context[:500]
        
        # Step 2: 답변 생성
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=config.SOLAR_TEMPERATURE,
                top_p=config.SOLAR_TOP_P,
                top_k=config.SOLAR_TOP_K,
                do_sample=True,
                repetition_penalty=config.SOLAR_REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return {
            "answer": answer,
            "context_used": context_used,
            "num_docs_retrieved": len(relevant_docs)
        }


if __name__ == "__main__":
    from vectordb_loader import FAISSVectorDBLoader
    
    # 벡터 DB 로딩
    vectordb = FAISSVectorDBLoader()
    
    # 답변 생성기 초기화
    generator = SOLARAnswerGenerator(vectordb)
    
    # 테스트
    test_question = "WMS 도입 비용은 얼마나 드나요?"
    result = generator.generate_answer(
        question=test_question,
        persona_name="김영수 (중소 물류센터 관리자)",
        persona_background="전통적인 수기 관리 방식에서 WMS 도입 검토 중"
    )
    
    print(f"\n질문: {test_question}")
    print(f"답변: {result['answer']}")
    print(f"검색된 문서: {result['num_docs_retrieved']}개")


"""EEVE 기반 WMS 답변 생성기 (통합 템플릿 + RAG)"""

import torch
from typing import Dict, List, Optional
from pathlib import Path
import sys

# 부모 디렉토리의 모듈 임포트를 위해
sys.path.append(str(Path(__file__).parent.parent / "Instruction"))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class PromptTemplate:
    """EEVE 통합 프롬프트 템플릿 (파인튜닝과 일관성 유지)"""
    
    # EEVE 공식 베이스 (파인튜닝과 동일)
    BASE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
    
    @staticmethod
    def wms_with_context(user_input: str, context: str, persona_info: str = "") -> str:
        """WMS 도메인 특화 (RAG 컨텍스트 포함)"""
        system_msg = f"""{PromptTemplate.BASE} The assistant is specialized in WMS (Warehouse Management System) and provides practical advice based on reference materials."""
        
        if persona_info:
            system_msg += f"\n\n[User Profile]\n{persona_info}"
        
        if context:
            system_msg += f"\n\n[Reference Materials]\n{context}"
        
        return f"""{system_msg}
Human: {user_input}
Assistant: """
    
    @staticmethod
    def wms_general(user_input: str, persona_info: str = "") -> str:
        """WMS 도메인 특화 (컨텍스트 없음)"""
        system_msg = f"""{PromptTemplate.BASE} The assistant is specialized in WMS (Warehouse Management System)."""
        
        if persona_info:
            system_msg += f"\n\n[User Profile]\n{persona_info}"
        
        return f"""{system_msg}
Human: {user_input}
Assistant: """


class GenerationConfig:
    """EEVE 생성 파라미터 (WMS 데이터 생성용)"""
    
    # TODO: WMS 데이터 품질에 맞게 조정 필요
    MAX_TOKENS = 600              # 최대 답변 길이
    TEMPERATURE = 0.8             # 다양성 (0.5-1.0 권장)
    TOP_P = 0.9                   # Nucleus sampling
    TOP_K = 50                    # Top-k sampling
    REPETITION_PENALTY = 1.15     # 반복 방지
    
    # RAG 설정
    TOP_K_DOCS = 3                # FAISS에서 가져올 문서 수
    MAX_CONTEXT_LENGTH = 1500     # 컨텍스트 최대 길이


class EEVEAnswerGenerator:
    """EEVE 기반 WMS 전문가 (VectorDB 검색 후 답변 생성)"""
    
    def __init__(
        self, 
        vectordb_loader,
        base_model_path: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",  # TODO: 베이스 모델 확인
        adapter_path: Optional[str] = None,  # TODO: 파인튜닝 완료 후 경로 설정
        use_4bit: bool = True
    ):
        """
        Args:
            vectordb_loader: FAISS 벡터 DB 로더
            base_model_path: EEVE 베이스 모델 경로
            adapter_path: LoRA 어댑터 경로 (옵션, 1단계 파인튜닝 후)
            use_4bit: 4bit 양자화 사용 여부
        """
        self.vectordb = vectordb_loader
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        
        print(f"\n{'='*60}")
        print(f"EEVE WMS 답변 생성기 초기화")
        print(f"{'='*60}")
        print(f"베이스 모델: {base_model_path}")
        if adapter_path:
            print(f"어댑터: {adapter_path}")
        else:
            print(f"어댑터: 없음 (베이스 모델만 사용)")
        print(f"{'='*60}\n")
        
        # 모델 로드
        self._load_model(use_4bit)
        
        print("✅ EEVE 모델 로딩 완료\n")
    
    def _load_model(self, use_4bit: bool):
        """모델 및 토크나이저 로드"""
        
        if use_4bit:
            # 4bit 양자화 (메모리 효율)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        else:
            # Full precision
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        
        # LoRA 어댑터 로드 (있다면)
        if self.adapter_path:
            from peft import PeftModel
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.adapter_path,
                    is_trainable=False
                )
                print("✓ LoRA 어댑터 로드 성공")
            except Exception as e:
                print(f"⚠️ LoRA 어댑터 로드 실패: {e}")
                print("베이스 모델만 사용합니다.")
        
        # 토크나이저
        tokenizer_path = self.adapter_path if self.adapter_path else self.base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def search_relevant_docs(self, question: str, k: int = None) -> List[str]:
        """FAISS에서 관련 문서 검색"""
        if k is None:
            k = GenerationConfig.TOP_K_DOCS
        
        try:
            return self.vectordb.search_relevant_docs(question, k=k)
        except Exception as e:
            print(f"⚠️ VectorDB 검색 실패: {e}")
            return []
    
    def generate_answer(
        self,
        question: str,
        persona_name: Optional[str] = None,
        persona_background: Optional[str] = None,
        use_rag: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> Dict:
        """
        WMS 질문에 대한 답변 생성 (RAG 기반)
        
        Args:
            question: 사용자 질문
            persona_name: 질문자 이름 (옵션)
            persona_background: 질문자 배경 (옵션)
            use_rag: RAG 사용 여부 (False면 일반 답변)
            max_tokens: 최대 생성 토큰 (None이면 기본값)
            temperature: 생성 temperature (None이면 기본값)
            top_p: Nucleus sampling (None이면 기본값)
            top_k: Top-k sampling (None이면 기본값)
            repetition_penalty: 반복 패널티 (None이면 기본값)
        
        Returns:
            Dict: {
                "answer": 생성된 답변,
                "context_used": 사용된 컨텍스트 (RAG 사용 시),
                "num_docs_retrieved": 검색된 문서 수,
                "prompt": 사용된 프롬프트 (디버깅용)
            }
        """
        
        # 파라미터 기본값
        if max_tokens is None:
            max_tokens = GenerationConfig.MAX_TOKENS
        if temperature is None:
            temperature = GenerationConfig.TEMPERATURE
        if top_p is None:
            top_p = GenerationConfig.TOP_P
        if top_k is None:
            top_k = GenerationConfig.TOP_K
        if repetition_penalty is None:
            repetition_penalty = GenerationConfig.REPETITION_PENALTY
        
        # Step 1: Persona 정보 구성
        persona_info = ""
        if persona_name and persona_background:
            persona_info = f"Name: {persona_name}\nBackground: {persona_background}"
        
        # Step 2: RAG 검색 (선택적)
        relevant_docs = []
        context_used = None
        
        if use_rag:
            relevant_docs = self.search_relevant_docs(question)
        
        # Step 3: 프롬프트 생성 (통합 템플릿 사용)
        if relevant_docs:
            # 컨텍스트 결합
            full_context = "\n\n".join(relevant_docs)
            context_trimmed = full_context[:GenerationConfig.MAX_CONTEXT_LENGTH]
            
            prompt = PromptTemplate.wms_with_context(
                question, 
                context_trimmed, 
                persona_info
            )
            context_used = context_trimmed[:500]  # 응답에 포함할 일부
        else:
            # RAG 없이 일반 답변
            prompt = PromptTemplate.wms_general(question, persona_info)
        
        # Step 4: 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        input_length = inputs.input_ids.shape[1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Step 5: 답변 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Step 6: 디코딩
        answer = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
        
        return {
            "answer": answer,
            "context_used": context_used,
            "num_docs_retrieved": len(relevant_docs),
            "prompt": prompt  # 디버깅용
        }


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("EEVE WMS 답변 생성기 테스트")
    print("="*60 + "\n")
    
    # TODO: FAISS VectorDB 로더 경로 확인
    try:
        from vectordb_loader import FAISSVectorDBLoader
        
        print("1. VectorDB 로딩 중...")
        vectordb = FAISSVectorDBLoader()
        print("   ✓ VectorDB 로드 완료\n")
        
    except Exception as e:
        print(f"⚠️ VectorDB 로드 실패: {e}")
        print("RAG 없이 테스트를 진행합니다.\n")
        vectordb = None
    
    # TODO: 모델 경로 설정
    # 옵션 1: 베이스 모델만 (1단계 파인튜닝 전)
    base_model = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    adapter = None
    
    # 옵션 2: 1단계 파인튜닝 완료 후
    # base_model = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    # adapter = "/home/work/eeve-korean-output/final"
    
    print("2. EEVE 모델 로딩 중...")
    generator = EEVEAnswerGenerator(
        vectordb_loader=vectordb,
        base_model_path=base_model,
        adapter_path=adapter,
        use_4bit=True
    )
    
    # TODO: 테스트 질문 수정
    test_cases = [
        {
            "question": "WMS 도입 비용은 얼마나 드나요?",
            "persona_name": "김영수",
            "persona_background": "중소 물류센터 관리자, 전통적인 수기 관리 방식에서 WMS 도입 검토 중"
        },
        {
            "question": "재고 실사 시간을 단축하려면 어떻게 해야 하나요?",
            "persona_name": "이민지",
            "persona_background": "대형 물류센터 재고 담당자, 월 1회 재고 실사에 3일 소요"
        },
        {
            "question": "바코드와 RFID 중 어떤 것을 선택해야 하나요?",
            "persona_name": "박준호",
            "persona_background": "IT 담당자, WMS 구축 기술 검토 중"
        }
    ]
    
    print("\n3. 답변 생성 테스트\n")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[테스트 케이스 {i}]")
        print(f"질문: {test['question']}")
        print(f"질문자: {test['persona_name']} ({test['persona_background'][:30]}...)")
        print("-" * 60)
        
        result = generator.generate_answer(
            question=test['question'],
            persona_name=test['persona_name'],
            persona_background=test['persona_background'],
            use_rag=(vectordb is not None)
        )
        
        print(f"답변: {result['answer'][:200]}...")
        print(f"검색 문서: {result['num_docs_retrieved']}개")
        if result['context_used']:
            print(f"컨텍스트: {result['context_used'][:100]}...")
        print("="*60)
    
    print("\n✅ 테스트 완료!")
    print("\n다음 단계:")
    print("1. TODO 주석 확인 및 경로 설정")
    print("2. 실제 WMS 질문으로 테스트")
    print("3. 생성 파라미터 조정 (temperature, top_p 등)")
    print("4. 대량 데이터셋 생성 (main.py 연동)")


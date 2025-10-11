#!/usr/bin/env python3
"""
EEVE 모델 대화 인터페이스
- EEVE 공식 프롬프트 템플릿 사용
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import argparse


class EEVEConversation:
    def __init__(
        self,
        base_model_name="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
        lora_model_path=None
    ):
        """
        EEVE 대화 시스템 초기화
        
        Args:
            base_model_name: 베이스 모델 (EEVE-10.8B)
            lora_model_path: LoRA 어댑터 경로 (옵션, 로컬 또는 HuggingFace)
        """
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """모델과 토크나이저 로드"""
        print(f"베이스 모델 로딩 중: {self.base_model_name}")
        if self.lora_model_path:
            print(f"LoRA 어댑터 로딩 중: {self.lora_model_path}")
        
        # 4-bit 양자화 설정 (메모리 효율)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # 베이스 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # LoRA 어댑터 로드 (있다면)
        if self.lora_model_path:
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_model_path,
                    is_trainable=False
                )
                print("✅ LoRA 어댑터 로드 완료")
            except Exception as e:
                print(f"⚠️ LoRA 어댑터 로드 실패: {e}")
                print("베이스 모델만 사용합니다.")
        
        # 토크나이저 로드
        if self.lora_model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.lora_model_path,
                    trust_remote_code=True
                )
                print("✅ 토크나이저 로드 (LoRA 경로)")
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True
                )
                print("✅ 토크나이저 로드 (베이스 모델)")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            print("✅ 토크나이저 로드 (베이스 모델)")
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("=" * 60)
        print("🚀 모델 로딩 완료! 대화를 시작하세요.")
        print(f"📝 Vocab 크기: {len(self.tokenizer):,}")
        print("=" * 60)
        
    def generate_response(
        self,
        user_input,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1
    ):
        """
        사용자 입력에 대한 응답 생성
        
        EEVE 최적 설정:
        - temperature=0.7 (적절한 다양성)
        - top_p=0.9 (자연스러운 sampling)
        - max_new_tokens=512 (8K 지원)
        
        Args:
            user_input: 사용자 메시지
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 (낮을수록 일관적)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: 반복 패널티
        """
        # EEVE 공식 프롬프트 템플릿
        prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """
        
        # 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        # 입력 길이 계산 (GPU 이동 전!)
        input_length = inputs.input_ids.shape[1]
        
        # GPU로 이동
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩 (입력 제외)
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self):
        """대화 루프"""
        print("\n💬 대화를 시작합니다. (종료: 'quit', 'exit', 'q')")
        print("📌 반말로 질문해도 존댓말로 답변합니다.")
        print("-" * 60)
        
        # 테스트 예시
        print("\n테스트 예시:")
        print("  - 한국의 수도가 어디야?")
        print("  - 파이썬 코드 짜줘")
        print("  - 피보나치 수열 설명해봐")
        print("-" * 60)
        
        while True:
            try:
                # 사용자 입력
                user_input = input("\n사용자: ").strip()
                
                # 종료 명령
                if user_input.lower() in ['quit', 'exit', 'q', '종료']:
                    print("\n👋 대화를 종료합니다.")
                    break
                
                # 빈 입력
                if not user_input:
                    continue
                
                # 응답 생성
                print("\n어시스턴트: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 대화를 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="EEVE 모델 대화 인터페이스"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
        help="베이스 모델 경로"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="LoRA 모델 경로 (옵션, 로컬 또는 HuggingFace)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="생성 temperature (기본: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="최대 생성 토큰 수 (기본: 512)"
    )
    
    args = parser.parse_args()
    
    # 대화 시스템 초기화
    conv = EEVEConversation(
        base_model_name=args.base_model,
        lora_model_path=args.model_path
    )
    
    # 모델 로드
    conv.load_model()
    
    # 대화 시작
    conv.chat()


if __name__ == "__main__":
    main()


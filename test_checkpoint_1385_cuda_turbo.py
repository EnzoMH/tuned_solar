#!/usr/bin/env python3
"""
SOLAR-10.7B checkpoint-1385 CUDA 터보 최적화 버전
데이터셋 전략 수립을 위한 고속 테스트
"""

import torch
import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings
import time
import gc
warnings.filterwarnings('ignore')

# CUDA 최적화 설정
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class CUDATurboSOLAR:
    def __init__(self, checkpoint_path="/home/work/tesseract/solar-korean-output/checkpoint-1385"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # CUDA 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
            
    def load_model_turbo(self):
        """CUDA 터보 모델 로딩"""
        print("🚀 CUDA 터보 로딩 시작...")
        print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        
        load_start = time.time()
        
        # 공격적인 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            "upstage/SOLAR-10.7B-v1.0",
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        # LoRA 어댑터 로드
        self.model = PeftModel.from_pretrained(
            base_model, 
            self.checkpoint_path,
            torch_dtype=torch.bfloat16
        )
        
        # 추론 최적화
        self.model.eval()
        
        # CUDA 최적화 설정
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True
            
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 첫 번째 더미 실행 (CUDA 워밍업)
        self._warmup()
        
        load_time = time.time() - load_start
        print(f"✅ 터보 로딩 완료! ({load_time:.1f}초)")
        
        return load_time
        
    def _warmup(self):
        """CUDA 캐시 워밍업"""
        print("🔥 CUDA 워밍업 중...")
        dummy_input = self.tokenizer("안녕", return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            _ = self.model.generate(
                **dummy_input,
                max_new_tokens=5,
                do_sample=False,
                use_cache=True
            )
        print("🔥 워밍업 완료!")
        
    def turbo_generate(self, question, max_tokens=50):
        """CUDA 터보 생성"""
        
        # 간결한 프롬프트
        prompt = f"질문: {question}\n답변:"
        
        # 토크나이징 (배치 처리 준비)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400,
            padding=False
        ).to(self.device)
        
        gen_start = time.time()
        
        # 터보 생성 (최적화된 파라미터)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Mixed precision
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.7,
                    top_k=25,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    use_cache=True,
                    num_beams=1,  # 빔 서치 비활성화
                )
        
        gen_time = time.time() - gen_start
        
        # 응답 추출
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 응답 정리
        if "답변:" in response:
            response = response.split("답변:")[-1].strip()
        if "질문:" in response:
            response = response.split("질문:")[0].strip()
            
        return response, gen_time
    
    def batch_generate(self, questions, max_tokens=50):
        """배치 처리로 여러 질문 동시 처리"""
        
        prompts = [f"질문: {q}\n답변:" for q in questions]
        
        # 배치 토크나이징
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=400,
            padding=True
        ).to(self.device)
        
        batch_start = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
        
        batch_time = time.time() - batch_start
        
        # 배치 응답 디코딩
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            if "답변:" in response:
                response = response.split("답변:")[-1].strip()
            responses.append(response)
            
        return responses, batch_time

def cuda_performance_test():
    """CUDA 성능 테스트"""
    
    print("🚀 SOLAR-1385 CUDA 터보 성능 테스트")
    print("="*60)
    
    turbo = CUDATurboSOLAR()
    load_time = turbo.load_model_turbo()
    
    # 개별 테스트
    print("\n🔸 개별 처리 테스트")
    questions = [
        "한국의 수도는 어디인가요?",
        "파이썬 프로그래밍이란 무엇인가요?",
        "김치찌개 만드는 방법을 알려주세요.",
        "인공지능의 정의는 무엇인가요?",
        "좋은 책을 추천해주세요."
    ]
    
    individual_times = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[테스트 {i}] {question}")
        
        try:
            answer, gen_time = turbo.turbo_generate(question, max_tokens=80)
            individual_times.append(gen_time)
            
            print(f"답변: {answer}")
            print(f"⏱️ 생성시간: {gen_time:.2f}초")
            
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 50)
    
    # 배치 테스트
    print("\n🔸 배치 처리 테스트")
    batch_questions = [
        "서울은?", "파이썬은?", "김치는?", "AI란?", "책 추천?"
    ]
    
    try:
        batch_answers, batch_time = turbo.batch_generate(batch_questions, max_tokens=30)
        
        print(f"배치 처리 결과 ({len(batch_questions)}개 질문):")
        for q, a in zip(batch_questions, batch_answers):
            print(f"  Q: {q} → A: {a}")
        print(f"⏱️ 배치 처리시간: {batch_time:.2f}초")
        print(f"📊 질문당 평균: {batch_time/len(batch_questions):.2f}초")
        
    except Exception as e:
        print(f"❌ 배치 처리 오류: {e}")
    
    # 성능 요약
    if individual_times:
        avg_individual = sum(individual_times) / len(individual_times)
        print(f"\n📊 성능 요약:")
        print(f"   모델 로딩: {load_time:.1f}초")
        print(f"   개별 평균: {avg_individual:.2f}초")
        print(f"   최고속도: {min(individual_times):.2f}초")
        print(f"   처리량: {1/avg_individual:.1f} 응답/초")

def interactive_turbo_chat():
    """터보 대화형 테스트"""
    
    turbo = CUDATurboSOLAR()
    turbo.load_model_turbo()
    
    print("\n🚀 CUDA 터보 채팅 (종료: 'quit')")
    print("🎯 데이터셋 전략 수립용 고속 테스트")
    print("="*50)
    
    while True:
        question = input("\n💭 질문: ").strip()
        
        if question.lower() in ['quit', '종료', 'q', 'exit']:
            print("🏁 터보 채팅 종료!")
            break
            
        if not question:
            continue
            
        start_time = time.time()
        
        try:
            answer, gen_time = turbo.turbo_generate(question, max_tokens=100)
            total_time = time.time() - start_time
            
            print(f"🤖 답변: {answer}")
            print(f"⚡ 처리시간: {total_time:.2f}초 (생성: {gen_time:.2f}초)")
            
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        interactive_turbo_chat()
    else:
        cuda_performance_test()

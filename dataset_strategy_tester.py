#!/usr/bin/env python3
"""
SOLAR checkpoint-1385 기반 데이터셋 전략 수립 도구
빠른 배치 처리로 다양한 데이터셋 패턴 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings
import time
import json
warnings.filterwarnings('ignore')

# CUDA 최적화
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class DatasetStrategyTester:
    def __init__(self, checkpoint_path="/home/work/tesseract/solar-korean-output/checkpoint-1385"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def load_model(self):
        """모델 로딩"""
        print("🚀 데이터셋 전략 테스터 로딩...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "upstage/SOLAR-10.7B-v1.0",
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.checkpoint_path)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 워밍업
        self._warmup()
        print("✅ 테스터 준비 완료!")
        
    def _warmup(self):
        """워밍업"""
        dummy = self.tokenizer("테스트", return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(**dummy, max_new_tokens=3, do_sample=False)
    
    def test_data_patterns(self):
        """다양한 데이터 패턴 테스트"""
        
        print("\n🎯 데이터셋 패턴별 성능 분석")
        print("="*60)
        
        # 데이터 패턴별 테스트 케이스
        test_patterns = {
            "단답형 QA": [
                "한국의 수도는?",
                "1+1은?", 
                "파이썬이란?",
                "김치는 무엇?",
                "AI 뜻은?"
            ],
            
            "설명형 QA": [
                "한국의 수도에 대해 설명해주세요",
                "파이썬 프로그래밍 언어를 설명해주세요",
                "김치찌개 만드는 방법을 알려주세요",
                "인공지능이 무엇인지 설명해주세요",
                "좋은 학습 방법을 설명해주세요"
            ],
            
            "대화형": [
                "안녕하세요! 오늘 기분이 어떠세요?",
                "요즘 뭘 하며 지내시나요?",
                "취미가 무엇인가요?",
                "좋아하는 음식이 있나요?",
                "주말에는 뭘 하시나요?"
            ],
            
            "전문적": [
                "머신러닝에서 오버피팅을 방지하는 방법은?",
                "파이썬에서 리스트와 튜플의 차이점은?",
                "딥러닝에서 경사하강법의 원리는?",
                "자연어처리에서 토큰화란 무엇인가?",
                "컴퓨터 비전에서 CNN의 역할은?"
            ],
            
            "창의적": [
                "미래의 AI 세상을 상상해서 이야기해주세요",
                "만약 하루가 48시간이라면 어떨까요?",
                "로봇과 인간이 함께 사는 세상을 그려보세요",
                "100년 후 기술은 어떻게 발전할까요?",
                "만약 시간여행이 가능하다면 어디로 가고싶나요?"
            ]
        }
        
        results = {}
        
        for pattern_name, questions in test_patterns.items():
            print(f"\n🔸 {pattern_name} 패턴 테스트")
            
            # 배치 처리로 빠르게 테스트
            answers, batch_time = self._batch_generate(questions, max_tokens=50)
            
            # 품질 분석
            quality_score = self._analyze_quality(questions, answers)
            
            results[pattern_name] = {
                "처리시간": batch_time,
                "평균시간": batch_time / len(questions),
                "품질점수": quality_score,
                "샘플": list(zip(questions[:2], answers[:2]))  # 샘플 2개만
            }
            
            print(f"   처리시간: {batch_time:.2f}초 ({batch_time/len(questions):.2f}초/질문)")
            print(f"   품질점수: {quality_score:.1f}/10")
            
            # 샘플 출력
            for q, a in list(zip(questions, answers))[:2]:
                print(f"   Q: {q}")
                print(f"   A: {a[:100]}..." if len(a) > 100 else f"   A: {a}")
                print()
        
        return results
    
    def _batch_generate(self, questions, max_tokens=50):
        """배치 생성"""
        prompts = [f"질문: {q}\n답변:" for q in questions]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=400,
            padding=True
        ).to(self.device)
        
        start_time = time.time()
        
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
        
        batch_time = time.time() - start_time
        
        # 응답 추출
        answers = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            if "답변:" in response:
                response = response.split("답변:")[-1].strip()
            answers.append(response)
            
        return answers, batch_time
    
    def _analyze_quality(self, questions, answers):
        """답변 품질 분석 (간단한 휴리스틱)"""
        total_score = 0
        
        for q, a in zip(questions, answers):
            score = 5  # 기본 점수
            
            # 길이 체크
            if len(a.strip()) < 5:
                score -= 3
            elif len(a.strip()) < 20:
                score -= 1
            elif len(a.strip()) > 200:
                score += 1
                
            # 한국어 비율 체크 (간단히)
            korean_chars = sum(1 for c in a if '\uac00' <= c <= '\ud7af')
            if len(a) > 0:
                korean_ratio = korean_chars / len(a)
                if korean_ratio > 0.3:
                    score += 2
                    
            # 반복 체크
            words = a.split()
            if len(words) > len(set(words)) * 1.5:
                score -= 2
                
            # URL이나 특수문자 체크
            if 'http' in a or '@@' in a or '##' in a:
                score -= 2
                
            total_score += max(0, min(10, score))
            
        return total_score / len(questions) if questions else 0
    
    def recommend_dataset_strategy(self, results):
        """데이터셋 구성 전략 추천"""
        
        print("\n🎯 데이터셋 구성 전략 추천")
        print("="*60)
        
        # 패턴별 성능 순위
        performance_ranking = sorted(
            results.items(), 
            key=lambda x: x[1]['품질점수'], 
            reverse=True
        )
        
        print("📊 패턴별 품질 순위:")
        for i, (pattern, data) in enumerate(performance_ranking, 1):
            print(f"   {i}. {pattern}: {data['품질점수']:.1f}점 ({data['평균시간']:.2f}초)")
        
        # 전략 추천
        best_pattern = performance_ranking[0][0]
        worst_pattern = performance_ranking[-1][0]
        
        print(f"\n💡 추천 전략:")
        print(f"   ✅ 강화할 패턴: {best_pattern} (현재 최고 성능)")
        print(f"   ⚠️  개선 필요: {worst_pattern} (추가 데이터 필요)")
        
        # 데이터셋 비율 추천
        total_quality = sum(data['품질점수'] for data in results.values())
        
        print(f"\n📋 권장 데이터셋 비율:")
        for pattern, data in results.items():
            ratio = (data['품질점수'] / total_quality) * 100 if total_quality > 0 else 20
            print(f"   {pattern}: {ratio:.0f}%")
            
        print(f"\n⚡ 처리 효율성:")
        avg_time = sum(data['평균시간'] for data in results.values()) / len(results)
        print(f"   평균 처리시간: {avg_time:.2f}초/질문")
        print(f"   배치 처리 권장: 5-10개씩 묶어서 처리")

def main():
    """메인 실행"""
    
    print("🎯 SOLAR-1385 데이터셋 전략 수립 도구")
    print("="*60)
    
    tester = DatasetStrategyTester()
    tester.load_model()
    
    # 패턴별 테스트 실행
    results = tester.test_data_patterns()
    
    # 전략 추천
    tester.recommend_dataset_strategy(results)
    
    # 결과 저장
    with open('/home/work/tesseract/dataset_strategy_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 분석 결과가 저장되었습니다: dataset_strategy_analysis.json")
    print("🎉 데이터셋 전략 분석 완료!")

if __name__ == "__main__":
    main()

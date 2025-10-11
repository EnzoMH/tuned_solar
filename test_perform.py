#!/usr/bin/env python3
"""
EEVE Checkpoint 성능 테스트
- 반말→존댓말 변환 확인
- 일반 한국어 질문 답변 품질 체크
- 다양한 시나리오 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import json

class CheckpointTester:
    """체크포인트 성능 테스트"""
    
    def __init__(
        self,
        base_model_path: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
        checkpoint_path: str = "/home/work/eeve-korean-output/checkpoint-500",  # TODO: 체크포인트 경로 확인
        use_4bit: bool = True
    ):
        self.base_model_path = base_model_path
        self.checkpoint_path = checkpoint_path
        
        print("\n" + "="*80)
        print(f" EEVE Checkpoint 성능 테스트")
        print("="*80)
        print(f"베이스 모델: {base_model_path}")
        print(f"체크포인트: {checkpoint_path}")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        self._load_model(use_4bit)
        
    def _load_model(self, use_4bit: bool):
        """모델 로드"""
        print("📥 모델 로딩 중...\n")
        
        if use_4bit:
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
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        
        print("   ✓ 베이스 모델 로드 완료")
        
        # LoRA 어댑터 로드
        self.model = PeftModel.from_pretrained(
            self.model,
            self.checkpoint_path,
            is_trainable=False
        )
        
        print("   ✓ 체크포인트 로드 완료")
        
        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("   ✓ 토크나이저 로드 완료\n")
    
    def generate(
        self,
        user_input: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> str:
        """답변 생성 (EEVE 프롬프트 템플릿)"""
        
        # EEVE 공식 프롬프트
        prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        input_length = inputs.input_ids.shape[1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
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
        
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def test_banmal_to_jondaemal(self):
        """반말→존댓말 변환 테스트"""
        print("📝 테스트 1: 반말 → 존댓말 변환")
        print("-" * 80)
        
        # TODO: 테스트 케이스 추가/수정 가능
        test_cases = [
            "한국의 수도가 어디야?",
            "피보나치 수열 설명해봐",
            "파이썬 코드 짜줘",
            "오늘 날씨 어때?",
            "맛있는 김치찌개 레시피 알려줘"
        ]
        
        results = []
        for i, question in enumerate(test_cases, 1):
            print(f"\n[{i}] 질문 (반말): {question}")
            
            response = self.generate(question, max_tokens=200)
            
            print(f"    답변: {response[:150]}{'...' if len(response) > 150 else ''}")
            
            # 존댓말 체크 (간단한 휴리스틱)
            jondaemal_markers = ['습니다', '입니다', '세요', '해요', '요.', '요!', '니다', '십시오']
            has_jondaemal = any(marker in response for marker in jondaemal_markers)
            
            print(f"    존댓말 사용: {'✅ YES' if has_jondaemal else '❌ NO'}")
            
            results.append({
                "question": question,
                "response": response,
                "has_jondaemal": has_jondaemal
            })
        
        success_rate = sum(1 for r in results if r['has_jondaemal']) / len(results) * 100
        print(f"\n✅ 존댓말 사용률: {success_rate:.0f}% ({sum(1 for r in results if r['has_jondaemal'])}/{len(results)})")
        
        return results
    
    def test_general_qa(self):
        """일반 QA 테스트"""
        print("\n\n📝 테스트 2: 일반 지식 질문")
        print("-" * 80)
        
        # TODO: 테스트 케이스 추가/수정 가능
        test_cases = [
            {
                "question": "인공지능이 뭐야?",
                "keywords": ["인공지능", "AI", "학습", "컴퓨터"]
            },
            {
                "question": "비트코인 설명해줘",
                "keywords": ["암호화폐", "블록체인", "디지털"]
            },
            {
                "question": "광합성이 뭐야?",
                "keywords": ["식물", "빛", "에너지", "산소"]
            },
            {
                "question": "WMS가 뭐야?",
                "keywords": ["WMS", "도입", "비용", "ROI"]
            },
            {
                "question": "WMS가 뭐야?",
                "keywords": ["WMS", "도입", "비용", "ROI"]
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            question = case['question']
            keywords = case['keywords']
            
            print(f"\n[{i}] 질문: {question}")
            
            response = self.generate(question, max_tokens=250)
            
            print(f"    답변: {response[:200]}{'...' if len(response) > 200 else ''}")
            
            # 키워드 포함 여부 체크
            keyword_found = [kw for kw in keywords if kw.lower() in response.lower()]
            
            print(f"    관련 키워드: {', '.join(keyword_found) if keyword_found else '없음'}")
            
            results.append({
                "question": question,
                "response": response,
                "keywords_found": keyword_found,
                "relevance_score": len(keyword_found) / len(keywords)
            })
        
        avg_relevance = sum(r['relevance_score'] for r in results) / len(results) * 100
        print(f"\n✅ 평균 관련성: {avg_relevance:.0f}%")
        
        return results
    
    def test_instruction_following(self):
        """지시 따르기 테스트"""
        print("\n\n📝 테스트 3: 지시 사항 이행")
        print("-" * 80)
        
        # TODO: 테스트 케이스 추가/수정 가능
        test_cases = [
            {
                "question": "3문장으로 한국 역사를 요약해줘",
                "check": "sentence_count",
                "target": 3
            },
            {
                "question": "5개의 프로그래밍 언어를 나열해줘",
                "check": "list_items",
                "keywords": ["Python", "Java", "JavaScript", "C", "C++", "Go", "Rust", "Ruby"]
            },
            {
                "question": "간단하게 한 줄로 설명해줘: 머신러닝이란?",
                "check": "length",
                "max_length": 100
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            question = case['question']
            check_type = case['check']
            
            print(f"\n[{i}] 질문: {question}")
            
            response = self.generate(question, max_tokens=200)
            
            print(f"    답변: {response}")
            
            # 검증
            passed = False
            if check_type == "sentence_count":
                sentence_count = response.count('.') + response.count('!') + response.count('?')
                passed = abs(sentence_count - case['target']) <= 2
                print(f"    문장 수: {sentence_count} (목표: {case['target']}) - {'✅' if passed else '❌'}")
            
            elif check_type == "list_items":
                items_found = [kw for kw in case['keywords'] if kw in response]
                passed = len(items_found) >= 3
                print(f"    항목 발견: {len(items_found)}개 ({', '.join(items_found[:5])}) - {'✅' if passed else '❌'}")
            
            elif check_type == "length":
                length = len(response)
                passed = length <= case['max_length']
                print(f"    길이: {length}자 (최대: {case['max_length']}자) - {'✅' if passed else '❌'}")
            
            results.append({
                "question": question,
                "response": response,
                "passed": passed
            })
        
        success_rate = sum(1 for r in results if r['passed']) / len(results) * 100
        print(f"\n✅ 지시 이행률: {success_rate:.0f}% ({sum(1 for r in results if r['passed'])}/{len(results)})")
        
        return results
    
    def test_creative_tasks(self):
        """창의적 작업 테스트"""
        print("\n\n📝 테스트 4: 창의적 작업")
        print("-" * 80)
        
        # TODO: 테스트 케이스 추가/수정 가능
        test_cases = [
            "짧은 시 하나 써줘",
            "재미있는 농담 하나 해줘",
            "간단한 Python 함수 예제 보여줘"
        ]
        
        results = []
        for i, question in enumerate(test_cases, 1):
            print(f"\n[{i}] 질문: {question}")
            
            response = self.generate(question, max_tokens=250, temperature=0.8)
            
            print(f"    답변: {response}")
            
            # 길이 체크 (너무 짧으면 실패)
            is_substantial = len(response) > 20
            print(f"    품질: {'✅ 충분함' if is_substantial else '❌ 너무 짧음'}")
            
            results.append({
                "question": question,
                "response": response,
                "is_substantial": is_substantial
            })
        
        return results
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        all_results = {}
        
        # 테스트 1: 반말→존댓말
        all_results['banmal_to_jondaemal'] = self.test_banmal_to_jondaemal()
        
        # 테스트 2: 일반 QA
        all_results['general_qa'] = self.test_general_qa()
        
        # 테스트 3: 지시 따르기
        all_results['instruction_following'] = self.test_instruction_following()
        
        # 테스트 4: 창의적 작업
        all_results['creative_tasks'] = self.test_creative_tasks()
        
        # 요약
        self.print_summary(all_results)
        
        return all_results
    
    # 테스트 결과 요약
    def print_summary(self, results):
        print("\n\n" + "="*80)
        print(" 📊 테스트 결과 요약")
        print("="*80)
        
        # 반말→존댓말
        jondaemal_rate = sum(1 for r in results['banmal_to_jondaemal'] if r['has_jondaemal']) / len(results['banmal_to_jondaemal']) * 100
        print(f"\n1. 반말→존댓말 변환: {jondaemal_rate:.0f}%")
        
        # 일반 QA
        qa_relevance = sum(r['relevance_score'] for r in results['general_qa']) / len(results['general_qa']) * 100
        print(f"2. 일반 QA 관련성: {qa_relevance:.0f}%")
        
        # 지시 따르기
        instruction_rate = sum(1 for r in results['instruction_following'] if r['passed']) / len(results['instruction_following']) * 100
        print(f"3. 지시 사항 이행: {instruction_rate:.0f}%")
        
        # 창의적 작업
        creative_rate = sum(1 for r in results['creative_tasks'] if r['is_substantial']) / len(results['creative_tasks']) * 100
        print(f"4. 창의적 작업: {creative_rate:.0f}%")
        
        # 전체 평균
        overall = (jondaemal_rate + qa_relevance + instruction_rate + creative_rate) / 4
        print(f"\n✨ 전체 평균 점수: {overall:.1f}%")
        
        print("\n" + "="*80)
        print(f"테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EEVE Checkpoint 성능 테스트")
    
    # TODO: 기본 경로 확인
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/work/eeve-korean-output/checkpoint-500",
        help="체크포인트 경로"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
        help="베이스 모델 경로"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        choices=['all', 'banmal', 'qa', 'instruction', 'creative'],
        default='all',
        help="실행할 테스트 (기본: all)"
    )
    
    args = parser.parse_args()
    
    # 테스터 초기화
    tester = CheckpointTester(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint
    )
    
    # 테스트 실행
    if args.test == 'all':
        results = tester.run_all_tests()
    elif args.test == 'banmal':
        results = tester.test_banmal_to_jondaemal()
    elif args.test == 'qa':
        results = tester.test_general_qa()
    elif args.test == 'instruction':
        results = tester.test_instruction_following()
    elif args.test == 'creative':
        results = tester.test_creative_tasks()
    
    # 결과 저장 (옵션)
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"📁 결과 저장: {output_file}")


if __name__ == "__main__":
    main()


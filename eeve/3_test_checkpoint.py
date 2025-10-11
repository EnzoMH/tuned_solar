#!/usr/bin/env python3
"""
EEVE 체크포인트 품질 테스트
- 저장된 체크포인트 로드
- 샘플 질문으로 답변 생성
- 품질 평가
"""

import torch
from unsloth import FastLanguageModel
import sys

def load_checkpoint(checkpoint_path):
    """체크포인트 로드"""
    print(f"\n{'='*80}")
    print(f"체크포인트 로딩: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True
    )
    
    # 추론 모드로 전환
    FastLanguageModel.for_inference(model)
    
    print("✓ 체크포인트 로드 완료\n")
    return model, tokenizer


def generate_response(model, tokenizer, question, max_new_tokens=512, temperature=0.7, repetition_penalty=1.2):
    """질문에 대한 답변 생성"""
    import time
    
    # EEVE 프롬프트 템플릿
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {question}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs['input_ids'].shape[1]
    
    # 추론 시작 시간
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=repetition_penalty,  # 반복 억제 (1.0~2.0, 높을수록 강함)
            no_repeat_ngram_size=3,  # 3-gram 반복 방지
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # 추론 종료 시간
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 생성된 토큰 수 계산
    generated_ids = outputs.sequences[0]
    total_tokens = len(generated_ids)
    generated_tokens = total_tokens - input_token_count
    
    # 초당 토큰 생성 속도
    tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0
    
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # "Assistant:" 이후 부분만 추출
    if "Assistant:" in response:
        answer = response.split("Assistant:")[-1].strip()
    else:
        answer = response
    
    # 통계 정보 반환
    stats = {
        "answer": answer,
        "input_tokens": input_token_count,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "inference_time": inference_time,
        "tokens_per_second": tokens_per_second
    }
    
    return stats


def run_test_suite(model, tokenizer):
    """테스트 질문 세트 실행"""
    
    test_questions = [
        {
            "category": "일반 상식",
            "question": "한국의 수도는 어디인가요?"
        },
        {
            "category": "설명 능력",
            "question": "양자역학을 초등학생도 이해할 수 있게 설명해주세요."
        },
        {
            "category": "창의성",
            "question": "AI가 인간을 대체할 수 있을까요? 찬반 양론을 설명해주세요."
        },
        {
            "category": "한국어 이해",
            "question": "'백지장도 맞들면 낫다'는 속담의 의미를 설명하고 실생활 예시를 들어주세요."
        },
        {
            "category": "코딩",
            "question": "Python으로 피보나치 수열을 구하는 함수를 작성해주세요."
        },
        {
            "category": "추론",
            "question": "철수는 영희보다 키가 크고, 영희는 민수보다 키가 큽니다. 셋 중 누가 가장 키가 큰가요?"
        }
    ]
    
    print(f"\n{'='*80}")
    print("테스트 시작")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, test in enumerate(test_questions, 1):
        print(f"\n[테스트 {idx}/{len(test_questions)}] {test['category']}")
        print(f"{'-'*80}")
        print(f"질문: {test['question']}")
        print(f"{'-'*80}")
        
        try:
            stats = generate_response(model, tokenizer, test['question'])
            print(f"답변: {stats['answer']}")
            print(f"\n[통계]")
            print(f"  입력 토큰: {stats['input_tokens']}")
            print(f"  생성 토큰: {stats['generated_tokens']}")
            print(f"  총 토큰: {stats['total_tokens']}")
            print(f"  추론 시간: {stats['inference_time']:.2f}초")
            print(f"  생성 속도: {stats['tokens_per_second']:.2f} tokens/sec")
            
            results.append({
                "category": test['category'],
                "question": test['question'],
                "answer": stats['answer'],
                "stats": stats,
                "success": True
            })
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "category": test['category'],
                "question": test['question'],
                "answer": None,
                "stats": None,
                "success": False
            })
        
        print(f"{'-'*80}")
    
    return results


def print_summary(results):
    """테스트 결과 요약"""
    print(f"\n\n{'='*80}")
    print("테스트 결과 요약")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"성공: {success_count}/{total_count}")
    print(f"성공률: {success_count/total_count*100:.1f}%\n")
    
    print("카테고리별 결과:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['category']}")
    
    # 전체 통계 계산
    successful_results = [r for r in results if r['success'] and r.get('stats')]
    if successful_results:
        total_input_tokens = sum(r['stats']['input_tokens'] for r in successful_results)
        total_generated_tokens = sum(r['stats']['generated_tokens'] for r in successful_results)
        total_time = sum(r['stats']['inference_time'] for r in successful_results)
        avg_time = total_time / len(successful_results)
        avg_tokens_per_sec = sum(r['stats']['tokens_per_second'] for r in successful_results) / len(successful_results)
        
        print(f"\n{'='*80}")
        print("전체 통계")
        print(f"{'='*80}")
        print(f"총 입력 토큰: {total_input_tokens:,}")
        print(f"총 생성 토큰: {total_generated_tokens:,}")
        print(f"총 추론 시간: {total_time:.2f}초")
        print(f"평균 추론 시간: {avg_time:.2f}초/질문")
        print(f"평균 생성 속도: {avg_tokens_per_sec:.2f} tokens/sec")
    
    print(f"\n{'='*80}")


def compare_checkpoints(checkpoint_paths):
    """여러 체크포인트 비교"""
    all_results = {}
    
    for checkpoint_path in checkpoint_paths:
        print(f"\n\n{'='*80}")
        print(f"체크포인트 테스트: {checkpoint_path}")
        print(f"{'='*80}")
        
        # 메모리 정리
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 체크포인트 로드
        model, tokenizer = load_checkpoint(checkpoint_path)
        
        # 테스트 실행
        results = run_test_suite(model, tokenizer)
        all_results[checkpoint_path] = results
        
        # 메모리 정리
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # 비교 결과 출력
    print(f"\n\n{'='*80}")
    print("체크포인트 비교 결과")
    print(f"{'='*80}\n")
    
    # 각 체크포인트별 요약
    for checkpoint_path, results in all_results.items():
        checkpoint_name = checkpoint_path.split('/')[-1]
        print(f"\n[{checkpoint_name}]")
        
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        print(f"  성공: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        successful_results = [r for r in results if r['success'] and r.get('stats')]
        if successful_results:
            avg_tokens_per_sec = sum(r['stats']['tokens_per_second'] for r in successful_results) / len(successful_results)
            avg_answer_length = sum(len(r['answer']) for r in successful_results) / len(successful_results)
            print(f"  평균 생성 속도: {avg_tokens_per_sec:.2f} tokens/sec")
            print(f"  평균 답변 길이: {avg_answer_length:.1f}자")
    
    # 질문별 답변 비교
    print(f"\n{'='*80}")
    print("질문별 답변 비교")
    print(f"{'='*80}\n")
    
    # 첫 번째 체크포인트의 질문 순서를 기준으로
    first_checkpoint = list(all_results.values())[0]
    
    for idx, test_result in enumerate(first_checkpoint, 1):
        print(f"\n[질문 {idx}] {test_result['category']}: {test_result['question']}")
        print(f"{'-'*80}")
        
        for checkpoint_path, results in all_results.items():
            checkpoint_name = checkpoint_path.split('/')[-1]
            result = results[idx-1]
            
            if result['success']:
                answer = result['answer']
                # 답변이 너무 길면 처음 200자만
                display_answer = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"\n[{checkpoint_name}]")
                print(f"{display_answer}")
                if result.get('stats'):
                    print(f"(생성: {result['stats']['generated_tokens']}토큰, {result['stats']['tokens_per_second']:.1f}tok/s)")
            else:
                print(f"\n[{checkpoint_name}]")
                print("오류 발생")
        
        print(f"{'-'*80}")
    
    print(f"\n{'='*80}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EEVE 체크포인트 품질 테스트')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/work/tesseract/eeve-korean-output-unsloth/checkpoint-2250',
        help='체크포인트 경로'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        default=None,
        help='비교할 체크포인트 경로들 (2개 이상)'
    )
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='단일 질문 테스트 (지정하지 않으면 전체 테스트 수행)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=512,
        help='최대 생성 토큰 수'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='샘플링 온도 (0.0~2.0)'
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.2,
        help='반복 억제 강도 (1.0~2.0, 높을수록 강함)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" EEVE 체크포인트 품질 테스트")
    print("="*80)
    
    # 비교 모드
    if args.compare:
        compare_checkpoints(args.compare)
        print("\n비교 테스트 완료!\n")
        return
    
    # 체크포인트 로드
    model, tokenizer = load_checkpoint(args.checkpoint)
    
    if args.question:
        # 단일 질문 테스트
        print(f"\n질문: {args.question}")
        print("-"*80)
        stats = generate_response(
            model, 
            tokenizer, 
            args.question,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty
        )
        print(f"답변: {stats['answer']}")
        print(f"\n[통계]")
        print(f"  입력 토큰: {stats['input_tokens']}")
        print(f"  생성 토큰: {stats['generated_tokens']}")
        print(f"  총 토큰: {stats['total_tokens']}")
        print(f"  추론 시간: {stats['inference_time']:.2f}초")
        print(f"  생성 속도: {stats['tokens_per_second']:.2f} tokens/sec")
        print("-"*80)
    else:
        # 전체 테스트 수행
        results = run_test_suite(model, tokenizer)
        print_summary(results)
    
    print("\n테스트 완료!\n")


if __name__ == "__main__":
    main()


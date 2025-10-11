#!/usr/bin/env python3
"""
LoRA 체크포인트를 베이스 모델과 병합 후 HuggingFace에 업로드
"""

import os
import sys
from unsloth import FastLanguageModel
from huggingface_hub import HfApi, login
import torch

def merge_lora_and_upload(
    checkpoint_path: str,
    base_model: str,
    hf_repo_id: str,
    hf_token: str = None,
    output_dir: str = None
):
    """
    LoRA 체크포인트를 베이스 모델과 병합 후 HuggingFace에 업로드
    
    Args:
        checkpoint_path: LoRA 체크포인트 경로
        base_model: 베이스 모델 이름
        hf_repo_id: HuggingFace 레포지토리 ID (예: "MyeongHo0621/eeve-vss-smh")
        hf_token: HuggingFace API 토큰 (없으면 환경변수에서 가져옴)
        output_dir: 로컬 저장 경로 (선택사항)
    """
    
    print(f"\n{'='*80}")
    print(" LoRA 병합 및 HuggingFace 업로드")
    print(f"{'='*80}\n")
    
    # HuggingFace 로그인
    if hf_token:
        login(token=hf_token)
    else:
        print("환경변수에서 HF_TOKEN 사용")
    
    # 1. 체크포인트 로드
    print(f"\n[1/4] 체크포인트 로딩: {checkpoint_path}")
    print(f"  베이스 모델: {base_model}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
        trust_remote_code=True
    )
    
    print("✓ 체크포인트 로드 완료")
    
    # 2. 병합 (16bit로)
    print(f"\n[2/4] LoRA 어댑터 병합 중...")
    print("  타입: float16")
    
    model = model.merge_and_unload()
    
    print("✓ 병합 완료")
    
    # 3. 로컬 저장 (선택사항)
    if output_dir:
        print(f"\n[3/4] 로컬 저장: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ 로컬 저장 완료: {output_dir}")
    else:
        print(f"\n[3/4] 로컬 저장 생략")
    
    # 4. HuggingFace 업로드
    print(f"\n[4/4] HuggingFace 업로드: {hf_repo_id}")
    
    model.push_to_hub(
        hf_repo_id,
        use_temp_dir=True,
        commit_message=f"Upload merged model from {checkpoint_path.split('/')[-1]}"
    )
    
    tokenizer.push_to_hub(
        hf_repo_id,
        use_temp_dir=True,
        commit_message=f"Upload tokenizer from {checkpoint_path.split('/')[-1]}"
    )
    
    print(f"✓ 업로드 완료: https://huggingface.co/{hf_repo_id}")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(" 완료!")
    print(f"{'='*80}\n")
    print(f"다음 단계:")
    print(f"1. https://huggingface.co/{hf_repo_id} 에서 확인")
    print(f"2. README.md 업데이트 (체크포인트 정보, 성능 등)")
    print(f"3. 모델 카드 작성 (훈련 세부사항, 예시 등)")
    print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LoRA 병합 및 HuggingFace 업로드')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='LoRA 체크포인트 경로'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='yanolja/EEVE-Korean-Instruct-10.8B-v1.0',
        help='베이스 모델 이름'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default='MyeongHo0621/eeve-vss-smh',
        help='HuggingFace 레포지토리 ID'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API 토큰 (없으면 환경변수 HF_TOKEN 사용)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='로컬 저장 경로 (선택사항)'
    )
    
    args = parser.parse_args()
    
    # 환경변수에서 토큰 가져오기
    if not args.token:
        args.token = os.environ.get('HF_TOKEN')
        if not args.token:
            print("경고: HF_TOKEN이 설정되지 않았습니다.")
            print("다음 중 하나를 수행하세요:")
            print("  1. --token 인자로 전달")
            print("  2. export HF_TOKEN=your_token")
            print("  3. .env 파일에 HF_TOKEN=your_token 추가")
            sys.exit(1)
    
    # 병합 및 업로드 실행
    merge_lora_and_upload(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        hf_repo_id=args.repo_id,
        hf_token=args.token,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()


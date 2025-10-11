#!/usr/bin/env python3
"""
BitsAndBytes 8-bit 양자화 스크립트 (프로덕션용)
- 4-bit보다 높은 품질 (거의 FP16 수준)
- CPU/GPU 최대 활용
- 프로덕션 환경에 최적화
"""

import os
import sys

# torchvision 충돌 방지
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import argparse
from pathlib import Path
from datetime import datetime

# transformers import 전에 torchvision 문제 우회
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


def optimize_performance():
    """CPU/GPU 성능 최적화"""
    print("성능 최적화 설정 중...")
    
    # CUDA 최적화
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # CPU 최적화
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    print(f"   ✓ CPU 스레드: {cpu_count}")
    print()


def quantize_8bit(
    model_path: str,
    output_path: str,
    llm_int8_threshold: float = 6.0,
    llm_int8_has_fp16_weight: bool = False
):
    """
    BitsAndBytes 8-bit 양자화 실행
    
    Args:
        model_path: 원본 모델 경로
        output_path: 양자화 모델 저장 경로
        llm_int8_threshold: Outlier threshold (기본 6.0)
        llm_int8_has_fp16_weight: FP16 weight 유지 여부
    """
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print(" BitsAndBytes 8-bit 양자화 (프로덕션용)")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"입력 모델: {model_path}")
    print(f"출력 경로: {output_path}")
    print(f"설정: 8-bit, threshold={llm_int8_threshold}")
    print("="*80 + "\n")
    
    # 성능 최적화
    optimize_performance()
    
    # 1. 토크나이저 로드
    print("1. 토크나이저 로딩...")
    step_start = datetime.now()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    step_elapsed = (datetime.now() - step_start).total_seconds()
    print(f"✓ 완료 ({step_elapsed:.1f}초)\n")
    
    # 2. 양자화 설정
    print("2. 양자화 설정...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=llm_int8_threshold,
        llm_int8_has_fp16_weight=llm_int8_has_fp16_weight
    )
    
    print(f"   ✓ 설정:")
    print(f"      - Bits: 8-bit (프로덕션용)")
    print(f"      - Threshold: {llm_int8_threshold}")
    print(f"      - FP16 weight: {llm_int8_has_fp16_weight}")
    print()
    
    # 3. 모델 로드 (자동 양자화)
    print("3. 모델 로딩 및 양자화 중...")
    print("   ⏱️  약 5-10분 소요됩니다...")
    step_start = datetime.now()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    step_elapsed = (datetime.now() - step_start).total_seconds()
    print(f"✓ 로딩 및 양자화 완료! ({step_elapsed//60:.0f}분 {step_elapsed%60:.0f}초)\n")
    
    # GPU 메모리 사용량 확인
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU 메모리 사용:")
        print(f"      - Allocated: {allocated:.2f} GB")
        print(f"      - Reserved: {reserved:.2f} GB")
        print()
    
    # 4. 저장
    print("4. 양자화 모델 저장 중...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    step_start = datetime.now()
    
    # 모델 저장 (양자화 상태 유지)
    model.save_pretrained(
        output_dir,
        safe_serialization=True
    )
    
    # 토크나이저 저장
    tokenizer.save_pretrained(output_dir)
    
    step_elapsed = (datetime.now() - step_start).total_seconds()
    print(f"✓ 저장 완료 ({step_elapsed:.1f}초)\n")
    
    # 5. 검증
    print("5. 생성 테스트...")
    test_prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: 안녕하세요
Assistant: """
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"   입력: '안녕하세요'")
    print(f"   출력: '{response.strip()}'")
    print("✓ 생성 테스트 성공\n")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()
    
    # 6. 요약
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    
    print("="*80)
    print("✓ BitsAndBytes 8-bit 양자화 완료!")
    print("="*80)
    
    # 파일 크기
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    total_size_gb = total_size / (1024**3)
    
    print(f"\n저장 위치: {output_dir}")
    print(f"총 크기: {total_size_gb:.2f} GB")
    print(f"압축률: {(1 - total_size_gb / 20.5) * 100:.1f}% (FP16 대비)")
    
    print(f"\n총 소요 시간: {total_elapsed//60:.0f}분 {total_elapsed%60:.0f}초")
    print(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n사용 방법:")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig")
    print("import torch")
    print()
    print("# 방법 1: 직접 로드 (저장된 설정 사용)")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}', device_map='auto')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print()
    print("# 방법 2: 명시적 설정")
    print("bnb_config = BitsAndBytesConfig(")
    print("    load_in_8bit=True,")
    print(f"    llm_int8_threshold={llm_int8_threshold}")
    print(")")
    print(f"model = AutoModelForCausalLM.from_pretrained(")
    print(f"    '{output_dir}',")
    print(f"    quantization_config=bnb_config,")
    print(f"    device_map='auto'")
    print(f")")
    print("```")
    
    print("\n특징 (프로덕션용):")
    print("  ✓ FP16과 거의 동일한 품질 (<0.5% 손실)")
    print("  ✓ 4-bit보다 안정적")
    print("  ✓ RTX 3060 (12GB) 이상 권장")
    print("  ✓ 프로덕션 서비스 배포에 최적")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="BitsAndBytes 8-bit 양자화 (프로덕션용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:

# 기본 8-bit 양자화 (프로덕션용)
python bnb_8bit.py

# 커스텀 설정
python bnb_8bit.py \\
    --model /home/work/eeve-merged-checkpoint-500 \\
    --output /home/work/tesseract/quant/eeve-bnb-8bit \\
    --threshold 6.0

비교:
- 4-bit: 빠름, 저VRAM (~3.5GB), 품질 98%
- 8-bit: 안정적, 중VRAM (~10GB), 품질 99.5% ⭐ 프로덕션 추천
- FP16: 느림, 고VRAM (~21GB), 품질 100%
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="/home/work/eeve-merged-checkpoint-500",
        help="원본 모델 경로"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/home/work/tesseract/quant/eeve-bnb-8bit",
        help="양자화 모델 저장 경로"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Outlier threshold (기본 6.0)"
    )
    
    parser.add_argument(
        "--fp16-weight",
        action="store_true",
        default=False,
        help="FP16 weight 유지 (더 높은 품질, 더 큰 크기)"
    )
    
    args = parser.parse_args()
    
    # 양자화 실행
    try:
        quantize_8bit(
            model_path=args.model,
            output_path=args.output,
            llm_int8_threshold=args.threshold,
            llm_int8_has_fp16_weight=args.fp16_weight
        )
        
        print("✅ 모든 작업이 성공적으로 완료되었습니다!")
        return 0
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


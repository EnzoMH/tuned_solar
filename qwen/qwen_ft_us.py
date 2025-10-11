#!/usr/bin/env python3
"""
Qwen 2.5-14B-Instruct Unsloth 고속 파인튜닝
- EEVE 대비 더 최신 모델 (2024년 9월)
- 중국어 베이스지만 한국어 성능 우수
- LoRA로 효율적 파인튜닝
- H100E 80GB 최적화
"""

import os
import torch
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QwenFineTuningConfig:
    """Qwen 2.5 파인튜닝 설정"""
    
    # 모델
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    max_seq_length: int = 4096  # Qwen 2.5는 최대 32K까지 지원
    
    # 데이터
    dataset_name: str = "MyeongHo0621/korean-quality-cleaned"
    
    # 출력
    output_dir: str = "/home/work/tesseract/qwen-korean-output-unsloth"
    run_name: str = f"qwen25-14b-unsloth-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA (H100E 80GB 최적화)
    lora_r: int = 128  # Rank
    lora_alpha: int = 256  # Alpha
    lora_dropout: float = 0.0  # Unsloth는 dropout=0 필수!
    
    # 훈련 설정 (14B는 7B보다 신중하게)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # 14B는 배치 4가 안전
    gradient_accumulation_steps: int = 4  # Effective batch = 16 유지
    learning_rate: float = 5e-5  # Qwen은 5e-05가 안정적
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"
    
    # 저장
    save_steps: int = 250
    save_total_limit: int = 5
    logging_steps: int = 10


class QwenFineTuner:
    """Qwen 2.5 Unsloth 기반 파인튜너"""
    
    def __init__(self, config: QwenFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Unsloth FastLanguageModel 로드"""
        logger.info("="*80)
        logger.info("Qwen 2.5-14B Unsloth 로딩")
        logger.info("="*80)
        logger.info(f"모델: {self.config.base_model}")
        logger.info(f"최대 시퀀스 길이: {self.config.max_seq_length}")
        logger.info("="*80)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # 자동 선택 (bfloat16)
            load_in_4bit=False,  # H100E는 full precision
            trust_remote_code=True
        )
        
        logger.info("✓ 모델 로드 완료")
        logger.info(f"✓ Tokenizer vocab: {len(self.tokenizer):,}")
        
        # LoRA 적용
        logger.info("="*80)
        logger.info("LoRA 설정 (Unsloth 최적화)")
        logger.info("="*80)
        logger.info(f"  r: {self.config.lora_r}")
        logger.info(f"  alpha: {self.config.lora_alpha}")
        logger.info(f"  dropout: {self.config.lora_dropout}")
        logger.info("="*80)
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=42,
            use_rslora=False,
            loftq_config=None
        )
        
        logger.info("✓ LoRA 적용 완료")
    
    def load_data(self):
        """데이터셋 로드"""
        logger.info("="*80)
        logger.info(f"데이터셋 로딩: {self.config.dataset_name}")
        logger.info("="*80)
        
        dataset = load_dataset(self.config.dataset_name, split="train")
        logger.info(f"✓ 전체 데이터: {len(dataset):,}개")
        
        # 통계
        sources = {}
        for example in dataset:
            source = example.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        logger.info("="*80)
        logger.info("데이터 소스 분포")
        logger.info("="*80)
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {source}: {count:,}개 ({count/len(dataset)*100:.1f}%)")
        logger.info("="*80)
        
        return dataset
    
    def format_prompts(self, examples):
        """Qwen 2.5 프롬프트 포맷팅 (ChatML 스타일)"""
        texts = []
        
        for messages in examples['messages']:
            # user와 assistant 추출
            user_content = None
            assistant_content = None
            
            if isinstance(messages, list) and len(messages) >= 2:
                for msg in messages:
                    if msg.get('role') == 'user' and user_content is None:
                        user_content = msg.get('content', '').strip()
                    elif msg.get('role') == 'assistant' and assistant_content is None:
                        assistant_content = msg.get('content', '').strip()
            
            # 유효한 데이터가 있으면 포맷팅
            if user_content and assistant_content:
                # Qwen 2.5 ChatML 템플릿
                text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
                texts.append(text)
            else:
                texts.append("")
        
        return {"text": texts}
    
    def train(self, dataset):
        """훈련 실행"""
        logger.info("="*80)
        logger.info(" Qwen 2.5-14B Unsloth 고속 파인튜닝 시작")
        logger.info(f" Run: {self.config.run_name}")
        logger.info("="*80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 데이터 포맷팅
        logger.info("데이터 포맷팅 중...")
        dataset = dataset.map(
            self.format_prompts,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 빈 문자열 필터링
        dataset = dataset.filter(lambda x: len(x['text']) > 0)
        logger.info(f"✓ 포맷팅 완료: {len(dataset):,}개")
        
        # 훈련/검증 분할
        train_size = int(0.95 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        logger.info(f"훈련: {len(train_dataset):,}개")
        logger.info(f"검증: {len(eval_dataset):,}개")
        logger.info(f"Epoch: {self.config.num_train_epochs}")
        logger.info(f"배치: {self.config.per_device_train_batch_size} × {self.config.gradient_accumulation_steps} = {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info("="*80)
        
        # 훈련 인자
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            run_name=self.config.run_name,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            
            # 최적화 (Unsloth)
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",  # Unsloth 8bit optimizer
            
            # 로깅/저장
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # 기타
            remove_unused_columns=False,
            report_to=[],
            seed=42,
            save_safetensors=True
        )
        
        # Trainer (SFTTrainer with Unsloth)
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args
        )
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 훈련 시작!")
        logger.info(" Qwen 2.5-14B: 최신 아키텍처 + 우수한 한국어 성능")
        logger.info(" Unsloth 최적화: 2-5배 속도 향상")
        logger.info(" 예상 시간: 4-6시간 (H100E 기준)")
        logger.info("="*80)
        
        # GPU 통계
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu_stats.name}")
        logger.info(f"VRAM 사용: {start_gpu_memory} GB / {max_memory} GB")
        logger.info("="*80)
        
        trainer.train()
        
        # 최종 저장
        final_path = os.path.join(self.config.output_dir, "final")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # GPU 통계
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        
        logger.info("="*80)
        logger.info(" ✅ 훈련 완료!")
        logger.info("="*80)
        logger.info(f"모델 저장: {final_path}")
        logger.info(f"Peak VRAM: {used_memory} GB")
        logger.info(f"Peak VRAM (LoRA): {used_memory_for_lora} GB")
        logger.info(f"VRAM 사용률: {used_percentage}%")
        logger.info("="*80)
        
        return final_path


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print(" Qwen 2.5-14B-Instruct Unsloth 고속 파인튜닝")
    print(" 최신 모델 + 우수한 한국어 성능")
    print("="*80)
    
    # 설정
    config = QwenFineTuningConfig()
    
    print(f"\n{'='*80}")
    print(" 설정 요약")
    print(f"{'='*80}")
    print(f"모델: {config.base_model}")
    print(f"데이터셋: {config.dataset_name}")
    print(f"출력: {config.output_dir}")
    print(f"최대 길이: {config.max_seq_length}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Epoch: {config.num_train_epochs}")
    print(f"배치: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"학습률: {config.learning_rate}")
    print(f"Optimizer: adamw_8bit (Unsloth)")
    print(f"{'='*80}\n")
    
    # 파인튜너
    finetuner = QwenFineTuner(config)
    
    # 실행
    try:
        # 1. 모델 로드 (Unsloth)
        finetuner.load_model()
        
        # 2. 데이터 로드
        dataset = finetuner.load_data()
        
        # 3. 훈련
        model_path = finetuner.train(dataset)
        
        print(f"\n{'='*80}")
        print(" 다음 단계")
        print(f"{'='*80}")
        print(f"1. 병합: 별도 스크립트 사용")
        print(f"2. 테스트: test_checkpoint.py --checkpoint {model_path}")
        print(f"3. vLLM 배포")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()


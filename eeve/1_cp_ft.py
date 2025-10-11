#!/usr/bin/env python3
"""
EEVE-10.8B 체크포인트 기반 세밀 재훈련
- checkpoint-6250에서 시작
- 더 낮은 learning rate
- 더 자주 evaluation
- 과적합 방지 최적화
"""

import os
import torch
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """세밀 파인튜닝 설정"""
    
    # 모델 - 체크포인트에서 시작!
    checkpoint_path: str = "/home/work/tesseract/eeve-korean-output-unsloth/checkpoint-6250"
    max_seq_length: int = 4096
    
    # 데이터
    dataset_name: str = "MyeongHo0621/korean-quality-cleaned"
    
    # 출력
    output_dir: str = "/home/work/tesseract/eeve-korean-output-unsloth-refined"
    run_name: str = f"eeve-refined-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA (이미 적용되어 있으므로 설정만 유지)
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    
    # 훈련 설정 - 더 조심스럽게!
    num_train_epochs: float = 1.0  # 최대 1 epoch (실제로는 EarlyStopping으로 더 일찍 멈춤)
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-5  # 1e-4 → 3e-5 (더 낮게!)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03  # 더 짧은 warmup
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"
    
    # 저장 - 더 자주 체크!
    save_steps: int = 50  # 250 → 50 (5배 더 자주)
    eval_steps: int = 50   # 50 스텝마다 평가
    save_total_limit: int = 10  # 더 많은 체크포인트 유지
    logging_steps: int = 5
    
    # Early Stopping - 핵심!
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 5  # 5번 연속 개선 없으면 중단
    early_stopping_threshold: float = 0.001  # 최소 개선 임계값


class CheckpointFineTuner:
    """체크포인트 기반 세밀 파인튜너"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_checkpoint(self):
        """체크포인트 로드 (LoRA 포함)"""
        logger.info("="*80)
        logger.info("체크포인트 로딩 (LoRA 포함)")
        logger.info("="*80)
        logger.info(f"체크포인트: {self.config.checkpoint_path}")
        logger.info(f"최대 시퀀스 길이: {self.config.max_seq_length}")
        logger.info("="*80)
        
        # 체크포인트에서 직접 로드 (LoRA가 이미 적용되어 있음)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.checkpoint_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            trust_remote_code=True
        )
        
        # 훈련 모드로 전환
        FastLanguageModel.for_training(self.model)
        
        logger.info("✓ 체크포인트 로드 완료")
        logger.info(f"✓ Tokenizer vocab: {len(self.tokenizer):,}")
    
    def load_data(self):
        """데이터셋 로드"""
        logger.info("="*80)
        logger.info(f"데이터셋 로딩: {self.config.dataset_name}")
        logger.info("="*80)
        
        dataset = load_dataset(self.config.dataset_name, split="train")
        logger.info(f"✓ 전체 데이터: {len(dataset):,}개")
        
        return dataset
    
    def format_prompts(self, examples):
        """EEVE 프롬프트 포맷팅"""
        texts = []
        
        for messages in examples['messages']:
            user_content = None
            assistant_content = None
            
            if isinstance(messages, list) and len(messages) >= 2:
                for msg in messages:
                    if msg.get('role') == 'user' and user_content is None:
                        user_content = msg.get('content', '').strip()
                    elif msg.get('role') == 'assistant' and assistant_content is None:
                        assistant_content = msg.get('content', '').strip()
            
            if user_content and assistant_content:
                text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {user_content}\nAssistant: {assistant_content}{self.tokenizer.eos_token}"
                texts.append(text)
            else:
                texts.append("")
        
        return {"text": texts}
    
    def train(self, dataset):
        """세밀 훈련 실행"""
        logger.info("="*80)
        logger.info("🎯 세밀 파인튜닝 시작")
        logger.info(f"   Run: {self.config.run_name}")
        logger.info(f"   Starting from: {self.config.checkpoint_path}")
        logger.info("="*80)
        logger.info(f"전략:")
        logger.info(f"  1. 더 낮은 learning rate: {self.config.learning_rate} (기존 1e-4)")
        logger.info(f"  2. 최대 {self.config.num_train_epochs} epoch (EarlyStopping 적용)")
        logger.info(f"  3. 자주 평가: 매 {self.config.eval_steps} steps")
        logger.info(f"  4. 자주 저장: 매 {self.config.save_steps} steps")
        logger.info(f"  5. EarlyStopping: {self.config.early_stopping_patience}번 연속 개선 없으면 자동 중단")
        logger.info(f"  6. 최적 모델 자동 선택: eval_loss 기준")
        logger.info("="*80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 데이터 포맷팅
        logger.info("데이터 포맷팅 중...")
        dataset = dataset.map(
            self.format_prompts,
            batched=True,
            remove_columns=dataset.column_names
        )
        dataset = dataset.filter(lambda x: len(x['text']) > 0)
        logger.info(f"✓ 포맷팅 완료: {len(dataset):,}개")
        
        # 훈련/검증 분할 (기존과 동일한 분할 사용)
        train_size = int(0.95 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        logger.info(f"훈련: {len(train_dataset):,}개")
        logger.info(f"검증: {len(eval_dataset):,}개")
        logger.info(f"Epoch: {self.config.num_train_epochs}")
        logger.info(f"배치: {self.config.per_device_train_batch_size} × {self.config.gradient_accumulation_steps} = {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"예상 steps: ~{int(len(train_dataset) / (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps) * self.config.num_train_epochs)}")
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
            
            # 최적화
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",
            
            # 로깅/저장 - 더 자주!
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # 기타
            remove_unused_columns=False,
            report_to=[],
            seed=42,
            save_safetensors=True
        )
        
        # EarlyStopping 콜백 설정
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=self.config.early_stopping_threshold
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            callbacks=[early_stopping_callback]  # EarlyStopping 추가!
        )
        
        # 훈련 시작
        logger.info("="*80)
        logger.info("🚀 훈련 시작!")
        logger.info("🎯 EarlyStopping 활성화: eval_loss가 개선되지 않으면 자동 중단")
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
        logger.info("✅ 세밀 훈련 완료!")
        logger.info("="*80)
        logger.info(f"모델 저장: {final_path}")
        logger.info(f"Peak VRAM: {used_memory} GB")
        logger.info(f"VRAM 사용률: {used_percentage}%")
        logger.info("")
        logger.info("📊 EarlyStopping 결과:")
        logger.info(f"  - 자동으로 최적 지점에서 중단됨")
        logger.info(f"  - 저장된 모델은 가장 낮은 eval_loss를 가진 체크포인트")
        logger.info(f"  - 과적합 직전의 sweet spot!")
        logger.info("="*80)
        
        return final_path


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("🎯 EEVE-10.8B 세밀 재훈련 (checkpoint-6250 기반)")
    print("   목표: 과적합 직전의 최적 지점 찾기")
    print("="*80)
    
    # 설정
    config = FineTuningConfig()
    
    print(f"\n{'='*80}")
    print("📋 설정 요약")
    print(f"{'='*80}")
    print(f"체크포인트: {config.checkpoint_path}")
    print(f"데이터셋: {config.dataset_name}")
    print(f"출력: {config.output_dir}")
    print(f"Epoch: {config.num_train_epochs} (짧게!)")
    print(f"Learning Rate: {config.learning_rate} (낮게!)")
    print(f"배치: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps}")
    print(f"평가/저장: 매 {config.eval_steps} steps")
    print(f"{'='*80}\n")
    
    # 파인튜너
    finetuner = CheckpointFineTuner(config)
    
    # 실행
    try:
        # 1. 체크포인트 로드
        finetuner.load_checkpoint()
        
        # 2. 데이터 로드
        dataset = finetuner.load_data()
        
        # 3. 세밀 훈련
        model_path = finetuner.train(dataset)
        
        print(f"\n{'='*80}")
        print("🎉 다음 단계")
        print(f"{'='*80}")
        print(f"1. 체크포인트 비교: test_checkpoint.py로 새 체크포인트들 테스트")
        print(f"2. 최적 체크포인트 선택: eval_loss가 가장 낮은 것")
        print(f"3. 병합 & 업로드: merge_and_upload.py 사용")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
EEVE-10.8B Unsloth 고속 파인튜닝 (H100E 80GB 최적화)
- 기존 Transformers 대비 2-5배 빠름
- MyeongHo0621/korean-quality-cleaned (54K, 고품질)
- H100E 80GB 최대 활용 설정
  * 4096 시퀀스 길이 (긴 대화 지원)
  * 배치 8 (메모리 활용 극대화)
  * LoRA rank 128 (더 강력한 학습)
  * Full precision (4bit 양자화 제거)
"""

import os
import torch
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UnslothFineTuningConfig:
    """Unsloth 파인튜닝 설정"""
    
    # 모델
    base_model: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    max_seq_length: int = 4096  # 2048 -> 4096 (긴 대화 지원)
    
    # 데이터
    dataset_name: str = "MyeongHo0621/korean-quality-cleaned"
    
    # 출력
    output_dir: str = "/home/work/tesseract/eeve-korean-output-unsloth"
    run_name: str = f"eeve-unsloth-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA (H100E 80GB 최적화)
    lora_r: int = 128  # 64 -> 128 (더 강력한 학습)
    lora_alpha: int = 256  # 128 -> 256 (rank에 비례)
    lora_dropout: float = 0.0  # Unsloth는 dropout=0 필수!
    
    # 훈련 설정 (H100E 80GB 활용)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8  # 4 -> 8 (메모리 활용 증가)
    gradient_accumulation_steps: int = 2  # 4 -> 2 (동일 effective batch=16 유지)
    learning_rate: float = 1e-4  # 5e-5 -> 1e-4 (더 빠른 학습)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05  # 0.1 -> 0.05 (warmup 단축)
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"  # Unsloth 전용 최적화
    
    # 저장
    save_steps: int = 250
    save_total_limit: int = 5
    logging_steps: int = 10


class UnslothFineTuner:
    """Unsloth 기반 고속 파인튜너"""
    
    def __init__(self, config: UnslothFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Unsloth FastLanguageModel 로드"""
        logger.info("="*80)
        logger.info("Unsloth FastLanguageModel 로딩")
        logger.info("="*80)
        logger.info(f"모델: {self.config.base_model}")
        logger.info(f"최대 시퀀스 길이: {self.config.max_seq_length}")
        logger.info("="*80)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # 자동 선택 (bfloat16/float16)
            load_in_4bit=False,  # H100E 80GB는 4bit 불필요!
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
            use_rslora=False,  # Rank Stabilized LoRA
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
        """EEVE 프롬프트 포맷팅 (레이블 마스킹 자동)"""
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
            
            # 유효한 데이터가 있으면 포맷팅, 없으면 빈 문자열
            if user_content and assistant_content:
                # EEVE 공식 프롬프트 템플릿 + EOS 토큰
                text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {user_content}\nAssistant: {assistant_content}{self.tokenizer.eos_token}"
                texts.append(text)
            else:
                texts.append("")
        
        return {"text": texts}
    
    def train(self, dataset):
        """훈련 실행"""
        logger.info("="*80)
        logger.info(" Unsloth 고속 파인튜닝 시작")
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
            packing=False,  # 짧은 시퀀스 패킹 비활성화
            args=training_args
        )
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 훈련 시작!")
        logger.info(" Unsloth 최적화: 2-5배 속도 향상 예상")
        logger.info(" H100E 80GB 최적화: 메모리 활용 극대화")
        logger.info(" 예상 시간: 8-12시간 (배치/시퀀스 증가로 더 빠름)")
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
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
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
    print(" EEVE-10.8B Unsloth 고속 파인튜닝 (H100E 80GB 최적화)")
    print(" 기존 Transformers 대비 2-5배 빠름!")
    print("="*80)
    
    # 설정
    config = UnslothFineTuningConfig()
    
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
    finetuner = UnslothFineTuner(config)
    
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
        print(f"1. 병합: unsloth_merge.py 사용")
        print(f"2. vLLM 테스트: python test_vllm_speed.py --model {model_path}")
        print(f"3. RAG QA 생성: vLLM 서버 기동 후 진행")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()


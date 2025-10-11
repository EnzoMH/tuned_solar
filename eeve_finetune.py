#!/usr/bin/env python3
"""
EEVE-10.8B 한국어 커스텀 Instruction 파인튜닝
- 이미 한국어 최적화되어 있어 Light CPT 불필요
- 반말 질문 → 존댓말 답변 스타일 학습
- H100E 최적화 (24코어, 192GB RAM, 1 FGPU)
"""

import os
import json
import torch
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EEVEFineTuningConfig:
    """EEVE 파인튜닝 설정 (H100E 최적화)"""
    
    # 모델 (이미 한국어 최적화 완료!)
    base_model: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    model_name: str = "EEVE-Custom-Instruct"
    
    # 데이터
    data_path: str = "/home/work/tesseract/korean_large_data/korean_large_dataset.json"
    max_samples: Optional[int] = 100000  # 54K-100K면 충분 (EEVE는 이미 잘 학습됨)
    
    # 출력
    output_dir: str = "/home/work/eeve-korean-output"
    
    # LoRA 설정 (가벼운 조정)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05  # 낮게 (과적합 방지, 이미 instruction-tuned)
    lora_target_modules: List[str] = None
    
    # 훈련 설정 (메모리 최적화)
    num_train_epochs: int = 2  # EEVE는 2면 충분
    per_device_train_batch_size: int = 8  # 메모리 절약 (8 → 4)
    gradient_accumulation_steps: int = 2  # 효과적 배치 = 16 (동일 유지)
    learning_rate: float = 1e-4  # 낮게 (이미 잘 학습됨)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 4096  # 메모리 절약 (4096 → 2048)
    
    # 양자화
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # 최적화 (H100E 활용)
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8  # 24코어 중 8개 활용
    bf16: bool = True  # H100E는 bf16 최적화
    
    # 저장
    save_steps: int = 250
    save_total_limit: int = 10
    logging_steps: int = 20
    eval_steps: int = 250
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # EEVE는 SOLAR 기반이므로 동일한 target modules
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class EEVEFineTuner:
    """EEVE 파인튜닝 트레이너"""
    
    def __init__(self, config: EEVEFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_data(self) -> Dataset:
        """로컬 데이터 로드 및 검증"""
        logger.info(f"데이터 로딩: {self.config.data_path}")
        
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"데이터 파일 없음: {self.config.data_path}")
        
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"원본 데이터: {len(data):,}개")
        
        # 데이터 검증
        valid_data = []
        for i, example in enumerate(data):
            if not isinstance(example, dict):
                continue
            
            messages = example.get('messages', [])
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            
            # user와 assistant 메시지 확인
            has_user = any(m.get('role') == 'user' for m in messages)
            has_assistant = any(m.get('role') == 'assistant' for m in messages)
            
            if not (has_user and has_assistant):
                continue
            
            valid_data.append(example)
        
        logger.info(f"유효 데이터: {len(valid_data):,}개")
        
        # 샘플링
        if self.config.max_samples and len(valid_data) > self.config.max_samples:
            import random
            random.seed(42)
            valid_data = random.sample(valid_data, self.config.max_samples)
            logger.info(f"샘플링: {len(valid_data):,}개")
        
        dataset = Dataset.from_list(valid_data)
        
        # 통계
        sources = {}
        for example in valid_data:
            source = example.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        logger.info("=== 데이터 소스 분포 ===")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {source}: {count:,}개 ({count/len(valid_data)*100:.1f}%)")
        
        return dataset
    
    def setup_model(self):
        """모델 및 토크나이저 설정"""
        logger.info(f"모델 로딩: {self.config.base_model}")
        logger.info("EEVE는 이미 한국어 vocabulary extension 완료 (40,960 vocab)")
        
        # 토크나이저 (EEVE 토크나이저 그대로 사용!)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"pad_token 설정: {self.tokenizer.eos_token}")
        
        logger.info(f"✓ 토크나이저 vocab 크기: {len(self.tokenizer):,}")
        
        # 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=self.config.use_nested_quant
        )
        
        # 베이스 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        logger.info("✓ 베이스 모델 로드 완료")
        
        # Gradient checkpointing 준비
        if self.config.gradient_checkpointing:
            self.model.enable_input_require_grads()
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=True
            )
            logger.info("✓ Gradient checkpointing 활성화")
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        logger.info("=== LoRA 설정 ===")
        logger.info(f"  r: {self.config.lora_r}")
        logger.info(f"  alpha: {self.config.lora_alpha}")
        logger.info(f"  dropout: {self.config.lora_dropout}")
        logger.info(f"  target_modules: {self.config.lora_target_modules}")
        
        # LoRA 적용
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 훈련 가능 파라미터 확인
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"✓ 훈련 가능: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        if trainable == 0:
            raise RuntimeError("훈련 가능한 파라미터가 없습니다!")
        
        self.model.train()
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """데이터셋 토크나이징 (EEVE 프롬프트 템플릿 사용)"""
        logger.info("데이터셋 전처리 시작...")
        logger.info("EEVE 공식 프롬프트 템플릿 사용")
        
        def tokenize_function(examples):
            """토크나이징 함수 (EEVE 프롬프트 템플릿 + 레이블 마스킹)"""
            batch_input_ids = []
            batch_attention_masks = []
            batch_labels = []
            
            for messages in examples['messages']:
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                
                # user와 assistant 메시지 추출
                user_content = None
                assistant_content = None
                
                for msg in messages:
                    if msg.get('role') == 'user' and user_content is None:
                        user_content = msg.get('content', '').strip()
                    elif msg.get('role') == 'assistant' and assistant_content is None:
                        assistant_content = msg.get('content', '').strip()
                
                if not user_content or not assistant_content:
                    continue
                
                # EEVE 공식 프롬프트 템플릿
                prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {user_content}\nAssistant: "
                response = f"{assistant_content}"
                full_text = prompt + response
                
                # 전체 토크나이징
                full_tokenized = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.config.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # 프롬프트만 토크나이징 (레이블 마스킹용)
                prompt_tokenized = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.config.max_length,
                    add_special_tokens=False
                )
                
                input_ids = full_tokenized["input_ids"].squeeze(0)
                attention_mask = full_tokenized["attention_mask"].squeeze(0)
                
                # 레이블 생성: 프롬프트 부분은 -100으로 마스킹
                labels = input_ids.clone()
                prompt_length = len(prompt_tokenized["input_ids"])
                labels[:prompt_length] = -100  # 프롬프트 부분 마스킹
                
                # padding 부분도 -100으로 마스킹
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_labels.append(labels)
            
            # 빈 배치 처리
            if not batch_input_ids:
                dummy_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: 안녕하세요\nAssistant: "
                dummy_response = "안녕하세요! 무엇을 도와드릴까요?"
                dummy_full = dummy_prompt + dummy_response
                
                dummy = self.tokenizer(
                    dummy_full,
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                dummy_labels = dummy["input_ids"].squeeze(0).clone()
                dummy_labels[:100] = -100  # 앞부분 마스킹
                
                return {
                    "input_ids": [dummy["input_ids"].squeeze(0)],
                    "attention_mask": [dummy["attention_mask"].squeeze(0)],
                    "labels": [dummy_labels]
                }
            
            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_masks,
                "labels": batch_labels
            }
        
        processed = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=32,
            num_proc=None,  # CUDA 충돌 방지
            remove_columns=dataset.column_names,
            desc="토크나이징 (EEVE 프롬프트)"
        )
        
        logger.info(f"✓ 전처리 완료: {len(processed):,}개")
        return processed
    
    def train(self, dataset: Dataset):
        """훈련 실행"""
        logger.info("=" * 80)
        logger.info(f" {self.config.model_name} 훈련 시작")
        logger.info("=" * 80)
        
        # 출력 디렉토리
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 훈련/검증 분할
        train_size = int(0.95 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        logger.info(f"훈련: {len(train_dataset):,}개")
        logger.info(f"검증: {len(eval_dataset):,}개")
        
        # 훈련 인자
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            
            # 최적화 (H100E)
            bf16=self.config.bf16,
            fp16=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # 로깅/저장
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # 기타
            remove_unused_columns=False,
            report_to=[],
            optim="adamw_torch_fused",
            max_grad_norm=1.0,
            save_safetensors=True,
            seed=42
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )
        
        # Forward pass 테스트
        logger.info("Forward pass 테스트...")
        try:
            sample_batch = next(iter(trainer.get_train_dataloader()))
            sample_batch = {k: v.to(self.model.device) for k, v in sample_batch.items()}
            
            with torch.no_grad():
                outputs = self.model(**sample_batch)
                logger.info(f"✓ Forward pass 성공! Loss: {outputs.loss:.4f}")
        except Exception as e:
            logger.error(f"❌ Forward pass 실패: {e}")
            raise
        
        # 훈련 시작
        logger.info("훈련 시작!")
        logger.info(f"예상 시간: 4-8시간 (H100E 기준)")
        trainer.train()
        
        # 최종 저장
        final_path = os.path.join(self.config.output_dir, "final")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info("=" * 80)
        logger.info(" ✅ 훈련 완료!")
        logger.info("=" * 80)
        logger.info(f"모델 저장 위치: {final_path}")
        
        return final_path


def main():
    """메인 함수"""
    print("=" * 80)
    print(" EEVE-10.8B 커스텀 Instruction 파인튜닝")
    print(" - 이미 한국어 최적화 완료 (Light CPT 불필요)")
    print(" - 반말 질문 → 존댓말 답변 스타일 학습")
    print("=" * 80)
    
    # 설정
    config = EEVEFineTuningConfig()
    
    print(f"\n모델: {config.base_model}")
    print(f"데이터: {config.data_path}")
    print(f"샘플링: {config.max_samples:,}개")
    print(f"출력: {config.output_dir}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"에포크: {config.num_train_epochs}")
    print(f"배치: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps}")
    print(f"학습률: {config.learning_rate}")
    print("-" * 80)
    
    # 파인튜너
    finetuner = EEVEFineTuner(config)
    
    # 실행
    try:
        # 1. 데이터 로드
        dataset = finetuner.load_data()
        
        # 2. 모델 설정
        finetuner.setup_model()
        
        # 3. 데이터 전처리
        processed_dataset = finetuner.process_dataset(dataset)
        
        # 4. 훈련
        model_path = finetuner.train(processed_dataset)
        
        print(f"\n다음 단계:")
        print(f"1. python conv_eeve.py --model-path {model_path}")
        print(f"2. 모델과 대화해보기")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()


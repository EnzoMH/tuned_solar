#!/usr/bin/env python3
"""
SOLAR-10.7B 한국어 파인튜닝 스크립트
KT Cloud H100E 환경 최적화 버전 (5시간 제한용)

환경 사양:
- GPU: H100E 80GB
- RAM: 192GiB
- CPU: 24 Cores
- PyTorch: 2.6
- Python: 3.12
- CUDA: 12.8
"""

import os
import json
import torch
import random
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Transformers and PEFT
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
    TaskType,
    PeftModel
)

# Datasets
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# KT Cloud H100E 최적화 Configuration (5시간 제한용)
# =============================================================================

@dataclass
class QLoRAConfig:
    # 모델
    base_model: str = "upstage/SOLAR-10.7B-v1.0"
    
    # 양자화
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # 훈련
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 2048
    output_dir: str = "./solar-qlora-output"

@dataclass
class KTCloudH100Config:
    """KT Cloud H100E 환경에 최적화된 설정 (5시간 제한)"""
    
    # Model settings
    base_model: str = "upstage/SOLAR-10.7B-v1.0"
    model_name: str = "SOLAR-10.7B-Korean-Instruct-Fast"
    
    # KT Cloud H100E 최적화 설정 (빠른 실행)
    output_dir: str = "/home/work/solar-korean-output"
    num_train_epochs: int = 3  # 16시간 활용: 100만 데이터 × 3에포크
    per_device_train_batch_size: int = 8   # Gradient checkpointing으로 메모리 절약
    gradient_accumulation_steps: int = 2   # 효과적 배치 크기 = 8 유지
    learning_rate: float = 1.5e-5  # 더 보수적인 학습률로 안정성 향상
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05  # warmup 증가로 안정성 향상
    max_length: int = 2048  # 최대출력길이
    
    # LoRA settings - SOLAR에 최적화 (16시간 최고 품질)
    lora_r: int = 64  # 16시간 활용으로 파라미터 4배 확대
    lora_alpha: int = 128  # r에 비례하여 조정
    lora_dropout: float = 0.1
    
    # Dataset settings - 16시간 최적화용
    max_samples: int = 300000  # 전체 데이터 중 30만개 사용 (최고 품질)
    
    # KT Cloud H100E 하드웨어 설정
    use_4bit: bool = True  # H100E는 메모리 충분
    use_bf16: bool = False   # H100E에서 BF16 최적
    use_flash_attention: bool = False  # fp16/bf16 호환성 문제로 비활성화
    device_map: str = "auto"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # H100E 성능 최적화
    dataloader_num_workers: int = 8  # 메모리 절약을 위해 축소
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = True  # enable_input_require_grads()로 호환성 해결
    torch_compile: bool = False  # 안정성 위해 비활성화
    
    # 저장 설정 (자주 저장)
    save_steps: int = 500
    save_total_limit: int = 10
    logging_steps: int = 50

config = KTCloudH100Config()

# =============================================================================
# KT Cloud H100E 환경 최적화 체크
# =============================================================================

def check_kt_cloud_environment():
    """KT Cloud H100E 환경 최적화 체크"""
    # Tokenizer 병렬 처리 경고 방지
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logger.info("KT Cloud H100E 환경 체크 시작")
    
    # PyTorch 버전 확인
    pytorch_version = torch.__version__
    logger.info(f"PyTorch 버전: {pytorch_version}")
    
    # CUDA 버전 확인
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        logger.info(f"CUDA 버전: {cuda_version}")
        
        # GPU 정보 확인
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            logger.info(f"GPU {i}: {props.name}, 메모리: {memory_gb:.1f}GB")
            
            # H100 확인
            if "H100" in props.name:
                logger.info("H100 감지! KT Cloud 최적화 모드 활성화")
    
    # CPU 정보
    cpu_count = os.cpu_count()
    logger.info(f"CPU 코어: {cpu_count}개")
    
    logger.info("KT Cloud 환경 체크 완료")
    return True

# =============================================================================
# 한국어 데이터 로더 (빠른 실행용)
# =============================================================================

class SOLARKoreanDataLoader:
    """SOLAR 한국어 데이터 로더 (빠른 실행용)"""
    
    def __init__(self, config: KTCloudH100Config):
        self.config = config
        
    def load_korean_datasets(self) -> Dataset:
        """한국어 데이터 로딩 (5시간 제한용)"""
        logger.info("🇰🇷 한국어 데이터 로딩 시작 (빠른 실행 모드)...")
        
        # 1. 사전 준비된 한국어 기본 데이터 로드 우선 시도
        try:
            data_path = "/home/work/tesseract/korean_data/korean_base_dataset.json"
            if os.path.exists(data_path):
                logger.info("사전 준비된 한국어 데이터 발견!")
                with open(data_path, "r", encoding="utf-8") as f:
                    korean_data = json.load(f)
                
                # 빠른 실행을 위해 샘플링
                if len(korean_data) > self.config.max_samples:
                    korean_data = random.sample(korean_data, self.config.max_samples)
                    logger.info(f"빠른 실행을 위해 {self.config.max_samples:,}개 샘플링")
                
                logger.info(f"한국어 기본 데이터 로드: {len(korean_data):,}개")
                return Dataset.from_list(korean_data)
                
        except Exception as e:
            logger.warning(f"사전 준비 데이터 로드 실패: {e}")
        
        # 2. 실시간 로드 (fallback)
        logger.info("실시간 한국어 데이터셋 로딩...")
        datasets_loaded = []
        
        # KoAlpaca - 빠른 로딩
        try:
            logger.info("KoAlpaca 로딩...")
            koalpaca = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
            koalpaca_sample = koalpaca.select(range(min(30000, len(koalpaca))))
            koalpaca_formatted = self._format_instruction_dataset(koalpaca_sample, "koalpaca")
            datasets_loaded.append(koalpaca_formatted)
            logger.info(f"KoAlpaca: {len(koalpaca_formatted):,}개")
        except Exception as e:
            logger.warning(f"KoAlpaca 로드 실패: {e}")
        
        # Ko-Ultrachat - 빠른 로딩
        try:
            logger.info("Ko-Ultrachat 로딩...")
            ultrachat = load_dataset("maywell/ko_Ultrachat_200k", split="train")
            ultrachat_sample = ultrachat.select(range(min(30000, len(ultrachat))))
            ultrachat_formatted = self._format_chat_dataset(ultrachat_sample)
            datasets_loaded.append(ultrachat_formatted)
            logger.info(f"Ko-Ultrachat: {len(ultrachat_formatted):,}개")
        except Exception as e:
            logger.warning(f"Ko-Ultrachat 로드 실패: {e}")
        
        if datasets_loaded:
            combined = concatenate_datasets(datasets_loaded)
            
            # 최대 샘플 수 제한
            if len(combined) > self.config.max_samples:
                indices = random.sample(range(len(combined)), self.config.max_samples)
                combined = combined.select(indices)
                logger.info(f"빠른 실행용 데이터: {len(combined):,}개")
                
            return combined
        else:
            logger.error("한국어 데이터 로드 완전 실패")
            return None
    
    def _format_instruction_dataset(self, dataset: Dataset, source: str) -> Dataset:
        """Instruction following 형식으로 변환"""
        def format_example(example):
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            
            if input_text and input_text.strip():
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction
                
            return {
                'messages': [
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': output}
                ],
                'source': source
            }
        
        return dataset.map(format_example, num_proc=None)  # 멀티프로세싱 완전 비활성화
    
    def _format_chat_dataset(self, dataset: Dataset) -> Dataset:
        """채팅 형식 데이터 처리"""
        def format_example(example):
            messages = example.get('messages', [])
            if isinstance(messages, list) and len(messages) >= 2:
                return {'messages': messages, 'source': 'ultrachat'}
            return None
        
        formatted = dataset.map(format_example, num_proc=None)  # 멀티프로세싱 완전 비활성화
        return formatted.filter(lambda x: x is not None)

# =============================================================================
# SOLAR 최적화 트레이너 (5시간 제한용)
# =============================================================================

class SOLARTrainer:
    """SOLAR-10.7B에 최적화된 KT Cloud H100E 트레이너 (빠른 실행)"""
    
    def __init__(self, config: KTCloudH100Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """SOLAR 모델 및 토크나이저 설정"""
        logger.info(f"SOLAR 모델 로딩: {self.config.base_model}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 1. BitsAndBytes 설정 추가
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # KT Cloud H100E 최적화 모델 로드 (메모리 최적화)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            trust_remote_code=True,
            use_cache=False,  # 훈련 시 메모리 절약
            low_cpu_mem_usage=True,  # CPU 메모리 사용량 최적화
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else None
        )
        
        logger.info("SOLAR 모델 로드 완료")
        
        # Gradient checkpointing과 LoRA 호환성 해결
        if self.config.gradient_checkpointing:
            self.model.enable_input_require_grads()
            logger.info("enable_input_require_grads() 적용 완료")
            
            # 추가 안정성을 위해 prepare_model_for_kbit_training도 적용
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=True
            )
            logger.info("prepare_model_for_kbit_training 적용 완료")
        
        # SOLAR에 최적화된 LoRA 설정 (안정성 우선)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=False  # 안정성 위해 비활성화
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 훈련 모드 설정
        self.model.train()
        
        # LoRA 파라미터 강제 활성화 (더 강력한 방법)
        logger.info("LoRA 파라미터 강제 활성화 시작...")
        lora_params_activated = 0
        
        for name, param in self.model.named_parameters():
            # LoRA 관련 파라미터 식별 및 강제 활성화
            if any(lora_key in name.lower() for lora_key in ['lora_', 'lora_a', 'lora_b']):
                param.requires_grad = True
                param.retain_grad()  # gradient를 유지
                lora_params_activated += 1
                logger.debug(f"LoRA 파라미터 활성화: {name}")
            else:
                # 베이스 모델 파라미터는 명시적으로 비활성화
                param.requires_grad = False
        
        logger.info(f"LoRA 파라미터 활성화 완료: {lora_params_activated}개")
        
        # 최종 검증: 훈련 가능한 파라미터 수 확인
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"훈련 가능 파라미터: {trainable_params:,}개 / 전체 {total_params:,}개")
        
        if trainable_params == 0:
            logger.error("훈련 가능한 파라미터가 없습니다!")
            raise RuntimeError("No trainable parameters found!")
        
        logger.info("SOLAR 모델 설정 완료")
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """SOLAR용 데이터셋 전처리 (노트북 방식 적용)"""
        logger.info("SOLAR용 데이터 전처리 시작...")
        
        def tokenize_function(examples):
            """노트북과 동일한 토크나이징 함수 (안전성 강화)"""
            batch_input_ids = []
            batch_attention_masks = []
            batch_labels = []
            
            for messages in examples['messages']:
                # None 값과 타입 체크
                if messages is None or not isinstance(messages, list):
                    continue
                
                # Chat template 적용 시도
                try:
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except:
                    # Chat template이 없으면 직접 포맷
                    formatted_text = ""
                    for msg in messages:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            if msg['role'] == 'user':
                                formatted_text += f"사용자: {msg['content']}\n"
                            elif msg['role'] == 'assistant':
                                formatted_text += f"Solar: {msg['content']}\n"
                
                # 빈 텍스트 건너뛰기
                if not formatted_text or not formatted_text.strip():
                    continue
                
                # 토크나이징
                tokenized = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                input_ids = tokenized["input_ids"].squeeze(0)
                attention_mask = tokenized["attention_mask"].squeeze(0)
                
                # 레이블 = input_ids (causal LM)
                labels = input_ids.clone()
                
                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_labels.append(labels)
            
            # 빈 배치 처리
            if not batch_input_ids:
                # 더미 데이터로 빈 배치 처리
                dummy_input = self.tokenizer(
                    "사용자: 안녕하세요\nSolar: 안녕하세요!",
                    truncation=True,
                    padding="max_length", 
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                return {
                    "input_ids": [dummy_input["input_ids"].squeeze(0)],
                    "attention_mask": [dummy_input["attention_mask"].squeeze(0)],
                    "labels": [dummy_input["input_ids"].squeeze(0)]
                }
            
            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_masks,
                "labels": batch_labels
            }
        
        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=32,
            num_proc=None,  # 완전히 멀티프로세싱 비활성화
            remove_columns=dataset.column_names
        )
        
        logger.info(f"전처리 완료: {len(processed_dataset)}개")
        return processed_dataset
    
    def train(self, dataset: Dataset):
        """KT Cloud H100E 최적화 훈련 (5시간 제한)"""
        logger.info(" KT Cloud H100E 최적화 훈련 시작 (빠른 실행 모드)...")
        
        # 출력 디렉토리 생성
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 훈련/검증 분할 (90%/10%)
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        logger.info(f"훈련 데이터: {len(train_dataset):,}개")
        logger.info(f"검증 데이터: {len(eval_dataset):,}개")
        
        # KT Cloud H100E 최적화 훈련 인자 (5시간 제한)
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            
            # H100 최적화
            bf16=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            
            # 로깅 및 저장 (자주 저장)
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # 성능 최적화
            remove_unused_columns=False,
            report_to=[],  # 로깅 도구 사용하지 않음
            logging_dir=f"{self.config.output_dir}/logs",
            
            # PyTorch 2.6 최적화
            optim="adamw_torch_fused",
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
            save_safetensors=True,
            seed=42
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        
        # 첫 번째 배치 테스트
        logger.info("첫 번째 배치 forward pass 테스트...")
        try:
            sample_batch = next(iter(trainer.get_train_dataloader()))
            sample_batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in sample_batch.items()}
            
            with torch.no_grad():
                outputs = self.model(**sample_batch)
                logger.info(f"Forward pass 성공! Loss: {outputs.loss:.4f}")
        except Exception as e:
            logger.error(f"Forward pass 실패: {e}")
            return None
        
        # 예상 시간 계산
        total_steps = len(train_dataset) // self.config.per_device_train_batch_size * self.config.num_train_epochs
        logger.info(f"총 스텝: {total_steps}")
        logger.info(f"예상 시간: 약 4-6시간 (H100, 배치크기 {self.config.per_device_train_batch_size}, 30만 샘플, 3 에포크)")  # 
        
        # 훈련 실행 (체크포인트부터 재개)
        try:
            # 마지막 체크포인트 찾기
            checkpoints = [d for d in os.listdir(self.config.output_dir) 
                          if d.startswith('checkpoint-')]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                resume_path = os.path.join(self.config.output_dir, last_checkpoint)
                logger.info(f"체크포인트부터 재개: {resume_path}")
                trainer.train(resume_from_checkpoint=resume_path)
            else:
                logger.info("새로운 훈련 시작")
                trainer.train()
            
            # 모델 저장
            final_path = os.path.join(self.config.output_dir, "final")
            trainer.save_model(final_path)
            self.tokenizer.save_pretrained(final_path)
            
            logger.info(f"훈련 완료! 모델 저장 위치: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"훈련 중 오류: {e}")
            return None

# =============================================================================
# 메인 실행 함수
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print(" SOLAR-10.7B 한국어 파인튜닝 (5시간 제한 버전)")
    print("=" * 80)
    print("목표: 빠른 한국어 능력 향상")
    print(f"데이터: 최대 {config.max_samples:,}개 샘플")
    print(f"예상 시간: 3-4시간")
    print(f"환경: KT Cloud H100E")
    print("-" * 80)
    
    start_time = datetime.now()
    logger.info(f" 파인튜닝 시작: {start_time}")
    
    # Step 1: 환경 체크
    logger.info("Step 1: KT Cloud H100E 환경 체크")
    if not check_kt_cloud_environment():
        logger.error("환경 체크 실패")
        return False
    
    # Step 2: 한국어 데이터 로드
    logger.info("Step 2: 한국어 데이터셋 준비")
    korean_loader = SOLARKoreanDataLoader(config)
    korean_dataset = korean_loader.load_korean_datasets()
    
    if korean_dataset is None:
        logger.error("데이터 로딩 실패")
        return False
    
    # Step 3: 모델 설정 및 훈련
    logger.info("Step 3: SOLAR 모델 훈련")
    trainer = SOLARTrainer(config)
    trainer.setup_model()
    
    processed_dataset = trainer.process_dataset(korean_dataset)
    model_path = trainer.train(processed_dataset)
    
    # 완료
    end_time = datetime.now()
    duration = end_time - start_time
    
    if model_path:
        logger.info("파인튜닝 성공!")
        logger.info(f"모델 위치: {model_path}")
        logger.info(f"총 소요시간: {duration}")
    else:
        logger.error("파인튜닝 실패")
    
    return model_path is not None

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
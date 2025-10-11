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
    max_samples: Optional[int] = 100000  # 54K-100K면 충분
    
    # 출력
    output_dir: str = "/home/work/eeve-korean-output"
    
    # LoRA 설정
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05  # 낮게 (과적합 방지)
    lora_target_modules: List[str] = None
    
    # 훈련 설정 (H100E 최적화)
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4  # 낮게 (이미 instruction-tuned)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 4096  # EEVE는 8K 지원하지만 4K로 시작
    
    # 양자화
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # 최적화 (H100E 활용)
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8  # 24코어 중 8개
    preprocessing_num_workers: int = 8  # 전처리용
    bf16: bool = True
    
    # 저장
    save_steps: int = 250
    save_total_limit: int = 10
    logging_steps: int = 20
    eval_steps: int = 250
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class SOLARFineTuningConfig:
    """SOLAR 파인튜닝 설정 (명확하고 검증 가능)"""
    
    # 모델
    base_model: str = "beomi/OPEN-SOLAR-KO-10.7B"  # 한국어 pre-trained 모델
    model_name: str = "OPEN-SOLAR-KO-Instruct"
    
    # 데이터
    data_path: str = "/home/work/tesseract/korean_large_data/korean_large_dataset.json"  # 191K 대규모 데이터
    max_samples: Optional[int] = None  # None이면 전체 사용
    
    # 출력
    output_dir: str = "/home/work/solar-korean-quality-output"
    
    # LoRA 설정 (명확하게)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 훈련 설정
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 효과적 배치 = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_length: int = 2048
    
    # 양자화
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # 최적화
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # 저장
    save_steps: int = 100
    save_total_limit: int = 20
    logging_steps: int = 20
    eval_steps: int = 100
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
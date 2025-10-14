#!/usr/bin/env python3
"""
Qwen 2.5-32B WMS Domain Fine-tuning (2단계)
- 1단계 훈련된 모델 불러오기
- WMS 도메인 데이터로 추가 미세조정
- EEVE 포맷 →  템플릿 변환
- Unsloth 고속 파인튜닝
"""

import os
import json
import torch
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QwenWMSConfig:
    """Qwen 2.5 WMS Domain Fine-tuning 설정 (2단계)"""
    
    # 1단계 훈련된 모델 경로
    stage1_model_path: str = "/home/work/tesseract/qwen-korean-fp8-output-unsloth/final"
    
    # 베이스 모델 (토크나이저용)
    base_model: str = "CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic"
    max_seq_length: int = 4096
    
    # 토크나이저 
    tokenizer_name: str = os.getenv("TOKENIZER")
    hf_token: Optional[str] = None
    
    # WMS 데이터셋
    dataset_path: str = "./dataset/wms_qa_dataset_v2_20000_eeve.jsonl"  # EEVE 포맷
    # dataset_path: str = "./dataset/checkpoint_20000_20251013_133415.jsonl"  # Messages 포맷
    
    # 출력
    output_dir: str = "/home/work/tesseract/qwen-wms-fp8-output-unsloth"
    run_name: str = f"qwen25-32b-wms-stage2-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA (2단계는 더 작은 rank 사용)
    lora_r: int = 64  # Stage 1: 128 → Stage 2: 64
    lora_alpha: int = 128  # Stage 1: 256 → Stage 2: 128
    lora_dropout: float = 0.0
    
    # 훈련 설정 (도메인 특화는 짧게)
    num_train_epochs: int = 2  # 도메인 데이터는 2 epoch 충분
    per_device_train_batch_size: int = 4  # 더 신중하게
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    learning_rate: float = 2e-5  # Stage 1보다 낮은 LR
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    # 최적화
    use_gradient_checkpointing: str = "unsloth"
    
    # 저장
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 20


class QwenWMSFineTuner:
    """Qwen WMS Domain Fine-tuner (2단계)"""
    
    def __init__(self, config: QwenWMSConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """1단계 훈련된 모델 + 토크나이저 로드"""
        logger.info("="*80)
        logger.info("Qwen 2.5-32B WMS Domain Fine-tuning (Stage 2)")
        logger.info("="*80)
        logger.info(f"Stage 1 모델: {self.config.stage1_model_path}")
        logger.info(f"토크나이저: {self.config.tokenizer_name}")
        logger.info(f"WMS 데이터: {self.config.dataset_path}")
        logger.info("="*80)
        
        # HF 토큰 설정
        hf_token = self.config.hf_token or os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("✓ HuggingFace 로그인 완료")
        
        # 1단계 모델 존재 확인
        if not os.path.exists(self.config.stage1_model_path):
            logger.warning(f"[ WARNING ] Stage 1 모델이 없습니다: {self.config.stage1_model_path}")
            logger.info(f"베이스 모델로 폴백: {self.config.base_model}")
            model_path = self.config.base_model
        else:
            logger.info(f"✓ Stage 1 모델 발견: {self.config.stage1_model_path}")
            model_path = self.config.stage1_model_path
        
        # 모델 로드
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            trust_remote_code=True
        )
        
        logger.info("✓ 모델 로드 완료")
        
        #   토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        logger.info(f"✓ 토크나이저 로드 완료")
        logger.info(f"✓ Tokenizer vocab: {len(self.tokenizer):,}")
        
        # 임베딩 크기 조정
        original_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(self.tokenizer)
        
        if original_vocab_size != new_vocab_size:
            logger.info(f" 임베딩 크기 조정: {original_vocab_size:,} → {new_vocab_size:,}")
            self.model.resize_token_embeddings(new_vocab_size)
            logger.info("✓ 임베딩 크기 조정 완료")
        
        # LoRA 적용 (2단계)
        logger.info("="*80)
        logger.info("LoRA 설정 (Stage 2: WMS Domain)")
        logger.info("="*80)
        logger.info(f"  r: {self.config.lora_r} (Stage 1: 128 → Stage 2: 64)")
        logger.info(f"  alpha: {self.config.lora_alpha} (Stage 1: 256 → Stage 2: 128)")
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
        """WMS 데이터셋 로드"""
        logger.info("="*80)
        logger.info(f"WMS 데이터셋 로딩: {self.config.dataset_path}")
        logger.info("="*80)
        
        # 파일 형식 확인
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            first_line = json.loads(f.readline())
        
        # EEVE 포맷인지 messages 포맷인지 확인
        if 'text' in first_line:
            logger.info("✓ EEVE 템플릿 포맷 감지")
            is_eeve_format = True
        elif 'messages' in first_line:
            logger.info("✓ OpenAI Messages 포맷 감지")
            is_eeve_format = False
        else:
            raise ValueError("Unknown dataset format")
        
        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")
        logger.info(f"✓ 전체 데이터: {len(dataset):,}개")
        
        # 포맷 변환
        if is_eeve_format:
            logger.info("EEVE → Messages 포맷 변환 중...")
            dataset = self.convert_eeve_to_messages(dataset)
        
        logger.info(f"✓ 최종 데이터: {len(dataset):,}개")
        
        # 통계
        if 'metadata' in dataset.column_names:
            topics = {}
            for example in dataset:
                topic = example.get('metadata', {}).get('topic', 'unknown')
                topics[topic] = topics.get(topic, 0) + 1
            
            logger.info("="*80)
            logger.info("데이터 분포 (상위 10개 토픽)")
            logger.info("="*80)
            for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {topic}: {count:,}개")
            logger.info("="*80)
        
        return dataset
    
    def convert_eeve_to_messages(self, dataset):
        """EEVE 템플릿 → OpenAI Messages 포맷 변환"""
        def parse_eeve(examples):
            all_messages = []
            
            for text in examples['text']:
                # EEVE 템플릿 파싱
                if 'Human: ' in text and 'Assistant: ' in text:
                    parts = text.split('Human: ')
                    if len(parts) > 1:
                        content = parts[1]
                        qa_parts = content.split('Assistant: ')
                        
                        if len(qa_parts) == 2:
                            question = qa_parts[0].strip()
                            answer = qa_parts[1].replace('<|im_end|>', '').strip()
                            
                            # Messages 포맷
                            messages = [
                                {"role": "system", "content": "당신은 10년 경력의 물류 시스템 전문가입니다. WMS에 대한 전문적이고 정확한 답변을 제공합니다."},
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                            all_messages.append(messages)
                        else:
                            all_messages.append([])
                    else:
                        all_messages.append([])
                else:
                    all_messages.append([])
            
            return {"messages": all_messages}
        
        dataset = dataset.map(
            parse_eeve,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 빈 messages 필터링
        dataset = dataset.filter(lambda x: len(x['messages']) > 0)
        
        return dataset
    
    def format_prompts(self, examples):
        """토크나이저로 프롬프트 포맷팅"""
        texts = []
        
        for messages in examples['messages']:
            # messages 검증
            if not isinstance(messages, list) or len(messages) < 2:
                texts.append("")
                continue
            
            # 템플릿 적용
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
            except Exception as e:
                logger.warning(f"템플릿 적용 실패: {e}")
                texts.append("")
        
        return {"text": texts}
    
    def train(self, dataset):
        """훈련 실행"""
        logger.info("="*80)
        logger.info(" Qwen 2.5-32B WMS Domain Fine-tuning 시작 (Stage 2)")
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
        logger.info(f"학습률: {self.config.learning_rate} (Stage 1보다 낮음)")
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
            args=training_args
        )
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 훈련 시작 (WMS Domain) ")
        logger.info(" Qwen 2.5-14B FP8: Stage 1 모델 기반")
        logger.info(" WMS Domain: 20K QA 샘플")
        logger.info(" 예상 시간: 2-3시간 (H100E 기준, 2 epoch)")
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
        logger.info(" [ OK ] WMS Domain Fine-tuning 완료!")
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
    print(" Qwen 2.5-32B WMS Domain Fine-tuning (Stage 2)")
    print(" Stage 1 모델 → WMS 도메인 특화")
    print("="*80)
    
    # 설정
    config = QwenWMSConfig()
    
    print(f"\n{'='*80}")
    print(" 설정 요약")
    print(f"{'='*80}")
    print(f"Stage 1 모델: {config.stage1_model_path}")
    print(f"토크나이저: {config.tokenizer_name}")
    print(f"WMS 데이터: {config.dataset_path}")
    print(f"출력: {config.output_dir}")
    print(f"최대 길이: {config.max_seq_length}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Epoch: {config.num_train_epochs} (도메인 특화)")
    print(f"배치: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"학습률: {config.learning_rate} (낮은 LR)")
    print(f"Optimizer: adamw_8bit (Unsloth)")
    print(f"{'='*80}\n")
    
    # 파인튜너
    finetuner = QwenWMSFineTuner(config)
    
    # 실행
    try:
        # 1. 모델 로드
        finetuner.load_model()
        
        # 2. 데이터 로드
        dataset = finetuner.load_data()
        
        # 3. 훈련
        model_path = finetuner.train(dataset)
        
        print(f"\n{'='*80}")
        print(" 다음 단계")
        print(f"{'='*80}")
        print(f"1. 병합: merge_lora.py --checkpoint {model_path}")
        print(f"2. 테스트: test_wms_model.py --checkpoint {model_path}")
        print(f"3. vLLM 배포")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()


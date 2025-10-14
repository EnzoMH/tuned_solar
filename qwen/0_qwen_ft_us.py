#!/usr/bin/env python3
"""
Qwen 2.5-14B-Instruct Unsloth 고속 파인튜닝
- EEVE 대비 더 최신 모델 (2024년 9월)
- INT8 양자화로 메모리 효율 + 고품질
- 중국어 베이스지만 한국어 성능 우수
- 한국어 최적화 토크나이저로 토큰 효율 향상
- LoRA로 효율적 파인튜닝
- H100E 80GB 최적화
"""

import os
import sys
import torch
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 모니터링 모듈
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'z_util'))
from cpu_mntrg import CPUMonitor
from gpu_mnrtg import GPUMonitor

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 글로벌 모니터
cpu_monitor = CPUMonitor()
gpu_monitor = GPUMonitor()

from dotenv import load_dotenv
load_dotenv()

os.getenv("TOKENIZER")

def log_system_resources(stage: str):
    """시스템 리소스 상태 로깅 (상세)"""
    logger.info("\n" + "="*80)
    logger.info(f"[{stage}] 시스템 리소스 상세 모니터링")
    logger.info("="*80)
    
    # CPU 상세
    cpu_monitor.log_snapshot(logger, stage)
    
    # GPU 상세
    if gpu_monitor.available:
        gpu_monitor.log_all_gpus(logger, stage)
        
        # 메모리 압박 확인
        for i in range(gpu_monitor.device_count):
            if not gpu_monitor.check_memory_available(i, required_gb=2.0):
                logger.warning(f"[ WARNING ]  GPU{i} 메모리 부족! (2GB 미만 남음)")
    
    # RAM 압박 확인
    if cpu_monitor.check_memory_pressure(threshold=85.0):
        logger.warning(f"[ WARNING ]  RAM 메모리 압박! (85% 이상 사용 중)")
    
    logger.info("="*80 + "\n")


class EmbeddingMonitorCallback(TrainerCallback):
    """임베딩 학습 모니터링 콜백"""
    
    def __init__(self, tokenizer, test_tokens=None):
        self.tokenizer = tokenizer
        self.test_tokens = test_tokens or ['데이터', '분석', '안녕', '감사']
        self.initial_embeddings = {}
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """학습 시작 시 초기 임베딩 저장"""
        if model is not None:
            embed_layer = model.get_input_embeddings()
            
            logger.info("="*80)
            logger.info("[ MONITOR ] 초기 임베딩 통계")
            logger.info("="*80)
            
            for token in self.test_tokens:
                try:
                    token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                    embedding = embed_layer.weight[token_id].detach().cpu().float().numpy()
                    self.initial_embeddings[token] = embedding
                    
                    logger.info(f"  {token}: 평균={embedding.mean():.6f}, 표준편차={embedding.std():.6f}")
                except Exception as e:
                    logger.warning(f"  {token}: 토큰화 실패 - {e}")
            
            logger.info("="*80)
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """평가 시 임베딩 변화 확인"""
        if model is not None and state.global_step > 0 and state.global_step % 500 == 0:
            embed_layer = model.get_input_embeddings()
            
            logger.info("="*80)
            logger.info(f"[ MONITOR ] Step {state.global_step} - 임베딩 변화")
            logger.info("="*80)
            
            # TensorBoard writer 가져오기
            tb_writer = None
            if hasattr(kwargs, 'get') and 'tb_writer' in kwargs:
                tb_writer = kwargs['tb_writer']
            
            for token in self.test_tokens:
                if token not in self.initial_embeddings:
                    continue
                    
                try:
                    token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                    current_emb = embed_layer.weight[token_id].detach().cpu().float().numpy()
                    initial_emb = self.initial_embeddings[token]
                    
                    # 변화량 계산
                    change = np.linalg.norm(current_emb - initial_emb)
                    similarity = np.dot(current_emb, initial_emb) / (
                        np.linalg.norm(current_emb) * np.linalg.norm(initial_emb) + 1e-8
                    )
                    
                    logger.info(f"  {token}:")
                    logger.info(f"    변화량: {change:.6f}")
                    logger.info(f"    초기 유사도: {similarity:.6f}")
                    logger.info(f"    현재 평균: {current_emb.mean():.6f}")
                except Exception as e:
                    logger.warning(f"  {token}: 계산 실패 - {e}")
            
            # '데이터' vs '분석' 유사도 (TensorBoard 로깅 포함)
            try:
                token1_id = self.tokenizer.encode('데이터', add_special_tokens=False)[0]
                token2_id = self.tokenizer.encode('분석', add_special_tokens=False)[0]
                
                emb1 = embed_layer.weight[token1_id].detach().cpu().float().numpy()
                emb2 = embed_layer.weight[token2_id].detach().cpu().float().numpy()
                
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
                )
                
                logger.info(f"\n  '데이터' ↔ '분석' 유사도: {similarity:.6f}")
                
                # ⭐ TensorBoard에 커스텀 메트릭 로깅
                if metrics is not None:
                    metrics['embedding/data_analysis_similarity'] = similarity
            except:
                pass
                
            logger.info("="*80)

@dataclass
class QwenFineTuningConfig:
    """Qwen 2.5-14B 파인튜닝 설정 """
    
    # 모델 (일반 버전 - Unsloth 호환)
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"  # 32B → 14B (속도 2배, 메모리 절감)
    max_seq_length: int = 4096  # Qwen 2.5는 최대 32K까지 지원
    
    # 토크나이저
    tokenizer_name: str = os.getenv("TOKENIZER")
    hf_token: Optional[str] = None  # Private repo 접근용 (환경변수에서 로드)
    
    # 데이터
    dataset_name: str = "MyeongHo0621/korean-quality-cleaned"
    
    # 출력
    output_dir: str = "/home/work/tesseract/qwen/qwen-KR-14B-output-unsloth"
    run_name: str = f"qwen25-14b-KR-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # LoRA (H100E 80GB 최적화)
    lora_r: int = 128  # Rank
    lora_alpha: int = 256  # Alpha
    lora_dropout: float = 0.0  # Unsloth는 dropout=0 필수!
    
    # 훈련 설정 (BF16 - 메모리 고려하여 배치 조정)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # BF16은 FP8보다 메모리 2배 사용
    gradient_accumulation_steps: int = 4  # Effective batch = 16 유지
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # 임베딩 학습을 위해 증가 (0.01 → 0.1)
    
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
        """Unsloth FastLanguageModel 로드 (INT8) + 한국어 최적화 토크나이저"""
        logger.info("="*80)
        logger.info("Qwen 2.5-14B + Korean-Optimized Tokenizer 로딩")
        logger.info("="*80)
        logger.info(f"모델: {self.config.base_model}")
        logger.info(f"토크나이저: {self.config.tokenizer_name}")
        logger.info(f"최대 시퀀스 길이: {self.config.max_seq_length}")
        logger.info(f"BF16 정밀도: 최고 품질 학습")
        logger.info("="*80)
        
        # HF 토큰 설정 (환경변수에서 로드)
        hf_token = self.config.hf_token or os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("[ COMPLETE ] HuggingFace 로그인 완료")
        
        # 1. Qwen 모델 로드 (기본 토크나이저와 함께)
        # FP8 모델의 weight_scale 파라미터 무시
        import warnings
        warnings.filterwarnings("ignore", message="Some weights.*were not used")
        
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            load_in_8bit=True,  # 14B INT8 양자화 (~14GB) - 4bit보다 품질 향상
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        
        # load_in_8bit=True 시: BitsAndBytes INT8 양자화
        # - 4bit(NF4)보다 품질 높음 (정확도 손실 거의 없음)
        # - 14B 모델: ~14GB VRAM (32B NF4: ~16GB와 비슷)
        # - Flash Attention 호환성 더 좋음
        
        logger.info("[ COMPLETE ] Qwen 모델 로드 완료")
        log_system_resources("모델 로드 후")
        
        # 2. 한국어 최적화 토크나이저로 교체
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        logger.info(f"[ COMPLETE ] 한국어 최적화 토크나이저 로드 완료")
        logger.info(f"  Vocab size: {len(self.tokenizer):,}")
        logger.info(f"  BOS token: {self.tokenizer.bos_token}")
        logger.info(f"  EOS token: {self.tokenizer.eos_token}")
        logger.info(f"  PAD token: {self.tokenizer.pad_token}")
        
        # PAD 토큰 설정 (없으면 EOS로 대체)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"  PAD token을 EOS로 설정: {self.tokenizer.pad_token}")
        
        # 3. 임베딩 크기 조정 (커스텀 토크나이저 vocabulary에 맞춤)
        original_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(self.tokenizer)
        
        if original_vocab_size != new_vocab_size:
            logger.info(f"[ ! ] Vocab 크기 불일치 감지!")
            logger.info(f"임베딩 크기 조정 중... {original_vocab_size:,} → {new_vocab_size:,}")
            
            # 임베딩 크기 조정
            self.model.resize_token_embeddings(new_vocab_size)
            
            logger.info("[ COMPLETE ] 임베딩 크기 조정 완료")
            logger.info(f"  새로운 임베딩 shape: {self.model.get_input_embeddings().weight.shape}")
            
            # 새 토큰 임베딩 초기화 (기존 평균값 사용)
            if new_vocab_size > original_vocab_size:
                logger.info(f"[ * ] 새 토큰 임베딩 초기화 중...")
                
                with torch.no_grad():
                    # 기존 임베딩 가져오기
                    embed_layer = self.model.get_input_embeddings()
                    old_embeddings = embed_layer.weight[:original_vocab_size]
                    
                    # 새 토큰 임베딩을 기존 평균으로 초기화
                    new_embeddings = embed_layer.weight[original_vocab_size:]
                    mean_embedding = old_embeddings.mean(dim=0, keepdim=True)
                    new_embeddings.copy_(mean_embedding.expand_as(new_embeddings))
                    
                    # 약간의 노이즈 추가 (다양성 확보)
                    noise = torch.randn_like(new_embeddings) * 0.02
                    new_embeddings.add_(noise)
                    
                    logger.info(f"  [ COMPLETE ] {new_vocab_size - original_vocab_size:,}개 토큰 초기화 완료")
                    logger.info(f"  초기화 방식: 기존 임베딩 평균 + 노이즈(σ=0.02)")
                    logger.info(f"  평균: {new_embeddings.mean():.6f}, 표준편차: {new_embeddings.std():.6f}")
            
            #  추가: 새 임베딩 초기화 확인
            embed_weight = self.model.get_input_embeddings().weight
            logger.info(f"  임베딩 통계:")
            logger.info(f"    평균: {embed_weight.mean():.6f}")
            logger.info(f"    표준편차: {embed_weight.std():.6f}")
        else:
            logger.info("[ COMPLETE ] Vocab 크기 일치 - 조정 불필요")
            
        logger.info("="*80)
        
        #  여기가 핵심 수정 부분! 
        # LoRA 적용
        logger.info("LoRA 설정 (Unsloth 최적화)")
        logger.info(f"  r: {self.config.lora_r}")
        logger.info(f"  alpha: {self.config.lora_alpha}")
        logger.info(f"  dropout: {self.config.lora_dropout}")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            #  이 부분 추가! 
            modules_to_save=["embed_tokens", "lm_head"],  # 임베딩과 출력층도 학습
            # 
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=42,
            use_rslora=False,
            loftq_config=None
        )
        
        logger.info("[ COMPLETE ] LoRA 적용 완료 (임베딩 & 출력층 포함)")  #  로그 메시지도 수정
        log_system_resources("LoRA 적용 후")
        logger.info("="*80)
    
    def load_data(self, streaming: bool = False):
        """
        데이터셋 로드
        
        Args:
            streaming: True면 스트리밍 모드 (메모리 효율적)
        """
        logger.info("="*80)
        logger.info(f"데이터셋 로딩: {self.config.dataset_name}")
        logger.info(f"  모드: {'스트리밍 (메모리 효율)' if streaming else '일반'}")
        logger.info("="*80)
        
        if streaming:
            # 스트리밍 모드: 메모리에 전체 로드 X
            dataset = load_dataset(
                self.config.dataset_name, 
                split="train",
                streaming=True
            )
            logger.info("[ COMPLETE ] 스트리밍 데이터셋 로드 완료")
            logger.info("  [ INFO ] 데이터를 필요할 때마다 가져옵니다 (메모리 절약)")
        else:
            # 일반 모드: 전체 데이터 메모리 로드
            dataset = load_dataset(self.config.dataset_name, split="train")
            logger.info(f"[ COMPLETE ] 전체 데이터: {len(dataset):,}개")
        
        return dataset
    
    def format_prompts(self, examples):
        """
        토크나이저 Chat Template 사용
        각 토크나이저는 자체 템플릿 형식을 가지고 있음
        """
        texts = []
        errors = 0
        
        for i, messages in enumerate(examples['messages']):
            # messages가 이미 올바른 형식인지 확인
            if not isinstance(messages, list):
                texts.append("")
                errors += 1
                continue
            
            # 토크나이저의 apply_chat_template 사용
            try:
                # system, user, assistant 역할 지원
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # 학습용: assistant 답변 포함
                )
                texts.append(formatted_text)
            except Exception as e:
                logger.warning(f"포맷팅 실패 (샘플 {i}): {e}")
                texts.append("")
                errors += 1
        
        if errors > 0:
            logger.warning(f"[ WARNING ]  배치에서 {errors}/{len(examples['messages'])} 샘플 포맷팅 실패")
        
        return {"text": texts}
    
    def format_for_training(self, examples):
        """
        배치 샘플 포맷팅 (on-the-fly)
        SFTTrainer의 formatting_func용 - 리스트 반환 필수!
        
        Args:
            examples: 배치 데이터 (dict with lists)
        
        Returns:
            List[str]: 포맷팅된 텍스트 리스트
        """
        formatted_texts = []
        
        # 배치로 들어온 경우 처리
        messages_list = examples.get('messages', [])
        
        # 단일 샘플인 경우 리스트로 변환
        if not isinstance(messages_list[0] if messages_list else None, list):
            messages_list = [messages_list]
        
        for messages in messages_list:
            if not isinstance(messages, list) or len(messages) == 0:
                formatted_texts.append("")
                continue
            
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)
            except Exception as e:
                logger.debug(f"포맷팅 실패: {e}")
                formatted_texts.append("")
        
        return formatted_texts
    
    def train(self, dataset):
        """훈련 실행 (스트리밍 지원)"""
        logger.info("="*80)
        logger.info(" Qwen 2.5-14B INT8 Unsloth 고속 파인튜닝 시작")
        logger.info(f" Run: {self.config.run_name}")
        logger.info("="*80)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # GPU 메모리 정리
        log_system_resources("학습 준비")
        if gpu_monitor.available:
            gpu_monitor.clear_cache()
            logger.info("[ COMPLETE ] GPU 캐시 정리 완료")
        
        # 데이터셋 타입 확인
        from datasets import IterableDataset, Dataset
        is_streaming = isinstance(dataset, IterableDataset)
        
        if is_streaming:
            # 스트리밍 데이터셋을 일반 데이터셋으로 변환 (Unsloth 호환성)
            logger.info("[ INFO ] 스트리밍 데이터셋 감지 - 메모리 효율적 변환")
            logger.info("  Unsloth가 스트리밍을 지원하지 않아 변환 필요")
            
            # 스트리밍에서 데이터 수집
            import itertools
            
            logger.info("  데이터 수집 중 (54,190개)...")
            all_examples = []
            for i, example in enumerate(itertools.islice(dataset, 54190)):
                all_examples.append(example)
                if (i + 1) % 10000 == 0:
                    logger.info(f"    진행: {i+1:,}/54,190")
                    # 메모리 압박 체크
                    if cpu_monitor.check_memory_pressure(threshold=90.0):
                        logger.warning(f"[ WARNING ] 메모리 압박! {i+1}개에서 중단")
                        break
            
            logger.info(f"[ COMPLETE ] {len(all_examples):,}개 수집")
            
            # 일반 데이터셋으로 변환
            dataset = Dataset.from_list(all_examples)
            del all_examples  # 메모리 해제
            
            logger.info("[ COMPLETE ] 데이터셋 변환 완료")
            log_system_resources("데이터셋 변환 후")
            
            # 훈련/검증 분할
            train_size = int(0.95 * len(dataset))
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            train_size = len(train_dataset)
            eval_size = len(eval_dataset)
            is_streaming = False  # 이제 일반 데이터셋
        else:
            # 일반 모드
            logger.info("[ INFO ] 일반 데이터셋으로 학습")
            
            # 훈련/검증 분할
            train_size = int(0.95 * len(dataset))
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            eval_size = len(dataset) - train_size
        
        logger.info(f"훈련: {train_size:,}개")
        logger.info(f"검증: {eval_size:,}개")
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
            report_to=[],  # TensorBoard 비활성화 (포트 접근 불가)
            # logging_dir=os.path.join(self.config.output_dir, "logs"),  # 불필요
            seed=42,
            save_safetensors=True
        )
        
        # Trainer (SFTTrainer with Unsloth)
        # 이제 모든 데이터셋이 일반 형식이므로 사전 포맷팅 필요
        logger.info("[ INFO ] 데이터 직접 포맷팅 시작 (pickle 문제 우회)")
        
        # 데이터 포맷팅
        log_system_resources("데이터 포맷팅 시작")
        
        try:
            logger.info(f"  훈련 데이터 직접 포맷팅 중... ({train_size:,}개)")
            
            # Python loop로 직접 포맷팅 (pickle 문제 우회)
            train_formatted = []
            for i, example in enumerate(train_dataset):
                try:
                    messages = example.get('messages', [])
                    if messages:
                        text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        train_formatted.append({"text": text})
                except Exception as e:
                    logger.debug(f"샘플 {i} 포맷팅 실패: {e}")
                
                # 진행 상황 로깅
                if (i + 1) % 10000 == 0:
                    logger.info(f"    진행: {i+1:,}/{train_size:,}")
            
            # Dataset 생성
            train_dataset = Dataset.from_list(train_formatted)
            del train_formatted
            logger.info(f"[ COMPLETE ] 훈련 데이터 포맷팅 완료: {len(train_dataset):,}개")
            
            logger.info(f"  검증 데이터 직접 포맷팅 중... ({eval_size:,}개)")
            eval_formatted = []
            for i, example in enumerate(eval_dataset):
                try:
                    messages = example.get('messages', [])
                    if messages:
                        text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        eval_formatted.append({"text": text})
                except Exception as e:
                    logger.debug(f"샘플 {i} 포맷팅 실패: {e}")
            
            eval_dataset = Dataset.from_list(eval_formatted)
            del eval_formatted
            logger.info(f"[ COMPLETE ] 검증 데이터 포맷팅 완료: {len(eval_dataset):,}개")
            
            log_system_resources("데이터 포맷팅 완료")
            
        except MemoryError as e:
            logger.error(f"[ FAIL ] 메모리 부족: {e}")
            log_system_resources("OOM 발생 시점")
            raise
        except Exception as e:
            logger.error(f"[ FAIL ] 데이터 포맷팅 실패: {e}")
            import traceback
            logger.error(f"  스택 트레이스:\n{traceback.format_exc()}")
            log_system_resources("에러 발생 시점")
            raise
        
        # Trainer 생성
        logger.info("[ INFO ] SFTTrainer 초기화 중...")
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
            callbacks=[EmbeddingMonitorCallback(self.tokenizer)]
        )
        logger.info("[ COMPLETE ] Trainer 초기화 완료")
        
        # 훈련 시작
        logger.info("="*80)
        logger.info(" 훈련 시작 ")
        logger.info(" Qwen 2.5-14B INT8: 최신 아키텍처")
        logger.info(" 한국어 최적화 토크나이저: 토큰 효율 10-20% 향상")
        logger.info(" INT8 양자화: 고품질 + 메모리 효율 (14GB VRAM)")
        logger.info(" Unsloth 최적화: 2-5배 속도 향상")
        logger.info(" 예상 시간: 1-1.5시간 (H100 × 2 기준, 14B 모델)")
        logger.info("="*80)
        
        # 훈련 시작 전 최종 리소스 체크
        log_system_resources("훈련 시작")
        
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
        logger.info(" [ OK ] 훈련 완료!")
        logger.info("="*80)
        logger.info(f"모델 저장: {final_path}")
        logger.info(f"Peak VRAM: {used_memory} GB")
        logger.info(f"Peak VRAM (LoRA): {used_memory_for_lora} GB")
        logger.info(f"VRAM 사용률: {used_percentage}%")
        logger.info("="*80)
        
        # 최종 임베딩 품질 검증
        logger.info("="*80)
        logger.info("[ VALIDATION ] 최종 임베딩 품질 검증")
        logger.info("="*80)
        
        embed_layer = self.model.get_input_embeddings()
        test_pairs = [
            ('데이터', '분석'),
            ('안녕', '감사'),
            ('인공', '지능'),
            ('재고', '관리'),
        ]
        
        for token1, token2 in test_pairs:
            try:
                id1 = self.tokenizer.encode(token1, add_special_tokens=False)[0]
                id2 = self.tokenizer.encode(token2, add_special_tokens=False)[0]
                
                emb1 = embed_layer.weight[id1].detach().cpu().float().numpy()
                emb2 = embed_layer.weight[id2].detach().cpu().float().numpy()
                
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                
                status = "[ V ]" if similarity > 0.3 else "[ X ]"
                logger.info(f"  '{token1}' ↔ '{token2}': {similarity:.4f} {status}")
            except Exception as e:
                logger.warning(f"  '{token1}' ↔ '{token2}': 계산 실패 - {e}")
        
        logger.info("="*80)
        logger.info("  기준: 유사도 > 0.3 권장, > 0.5 우수, > 0.7 매우 우수")
        logger.info("="*80)
        
        return final_path


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print(" Qwen 2.5-14B-Instruct + Korean-Optimized Tokenizer")
    print(" INT8 양자화 + 한국어 최적화 토크나이저 + 우수한 한국어 성능")
    print("="*80)
    
    # 설정
    config = QwenFineTuningConfig()
    
    print(f"\n{'='*80}")
    print(" 설정 요약")
    print(f"{'='*80}")
    print(f"모델: {config.base_model}")
    print(f"토크나이저: {config.tokenizer_name}")
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
        
        # 2. 데이터 로드 (스트리밍 모드로 메모리 절약)
        dataset = finetuner.load_data(streaming=True)
        
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


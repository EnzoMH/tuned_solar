# Unsloth 고속 파인튜닝 가이드

## 왜 Unsloth인가?

### 속도 비교

| 방법 | 속도/step | 총 시간 (9,654 steps) | 가속비 |
|------|-----------|----------------------|--------|
| **Transformers + PEFT** | 10.97초 | ~29시간 | 1x |
| **Unsloth** | 4-6초 | **10-16시간** | **2-5배** |

### Unsloth 최적화

1. **Flash Attention 2** - Attention 연산 최적화
2. **RoPE Scaling** - 위치 임베딩 최적화
3. **Triton Kernels** - 커스텀 CUDA 커널
4. **8-bit Optimizer** - 메모리 효율적인 옵티마이저
5. **Gradient Checkpointing** - Unsloth 전용 최적화

---

## 즉시 실행

### 1. 기본 실행 (권장)

```bash
cd /home/work/tesseract

# 직접 실행
python eeve/eeve_finetune_unsloth.py

# 또는 nohup으로 백그라운드 실행
nohup python eeve/eeve_finetune_unsloth.py > eeve/training_unsloth.log 2>&1 &

# 로그 확인
tail -f eeve/training_unsloth.log
```

### 2. tmux로 실행

```bash
# tmux 세션 생성
tmux new -s eeve_unsloth

# 훈련 시작
cd /home/work/tesseract
python eeve/eeve_finetune_unsloth.py

# Detach: Ctrl+B, D
# 재접속: tmux attach -t eeve_unsloth
```

---

## 주요 변경 사항

### Transformers → Unsloth

| 항목 | Transformers | Unsloth |
|------|-------------|---------|
| **모델 로드** | `AutoModelForCausalLM` | `FastLanguageModel` |
| **LoRA 적용** | `get_peft_model()` | `FastLanguageModel.get_peft_model()` |
| **Trainer** | `Trainer` | `SFTTrainer` |
| **Optimizer** | `adamw_torch_fused` | `adamw_8bit` |
| **Gradient Checkpointing** | `True` | `"unsloth"` |

### 자동 최적화

- ✅ **자동 dtype 선택** (bfloat16/float16)
- ✅ **자동 Flash Attention 2**
- ✅ **자동 RoPE Scaling**
- ✅ **레이블 마스킹** (SFTTrainer 자동 처리)

---

## 설정 상세

### 모델 설정

```python
base_model = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
max_seq_length = 2048
load_in_4bit = True
```

### 데이터 설정

```python
dataset_name = "MyeongHo0621/korean-quality-cleaned"
# 54,190개 고품질 한국어 데이터
```

### LoRA 설정

```python
lora_r = 64
lora_alpha = 128
lora_dropout = 0.1
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### 훈련 설정

```python
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4  # 효과적 배치 = 16
learning_rate = 5e-5
warmup_ratio = 0.1
```

### 최적화 설정

```python
optimizer = "adamw_8bit"  # Unsloth 8-bit optimizer
use_gradient_checkpointing = "unsloth"  # Unsloth 전용
bf16 = True (if supported)
```

---

## 출력 구조

```
/home/work/tesseract/eeve-korean-output-unsloth/
├── checkpoint-250/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── checkpoint-500/
├── checkpoint-750/
├── ...
└── final/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── config.json
    └── tokenizer files
```

---

## 훈련 모니터링

### 로그 확인

```bash
# 실시간 로그
tail -f eeve/training_unsloth.log

# loss 추출
grep "loss" eeve/training_unsloth.log

# GPU 사용량 확인
watch -n 1 nvidia-smi
```

### 예상 진행 상황

```
Step 10:   loss ~1.38, 4-6초/step
Step 100:  loss ~1.35, 안정화
Step 500:  loss ~1.2-1.3
Step 1000: loss ~1.1-1.2
...
최종:      loss ~0.9-1.0 (목표)
```

---

## 훈련 완료 후

### 1. LoRA 병합

```bash
python -c "
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='/home/work/tesseract/eeve-korean-output-unsloth/final',
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False
)

# 병합
model.save_pretrained_merged(
    '/home/work/tesseract/eeve-unsloth-merged',
    tokenizer,
    save_method='merged_16bit'
)
print('✓ 병합 완료!')
"
```

### 2. vLLM 테스트

```bash
python test_vllm_speed.py \
  --model /home/work/tesseract/eeve-unsloth-merged \
  --repetition-penalty 1.15
```

### 3. Hugging Face 업로드

```bash
huggingface-cli login
huggingface-cli upload \
  MyeongHo0621/eeve-vss-smh-v2 \
  /home/work/tesseract/eeve-unsloth-merged
```

---

## 문제 해결

### OOM (Out of Memory)

```python
# eeve_finetune_unsloth.py에서 수정
per_device_train_batch_size = 2  # 4 → 2
gradient_accumulation_steps = 8  # 4 → 8
```

### 속도가 느린 경우

```python
# max_seq_length 줄이기
max_seq_length = 1024  # 2048 → 1024
```

### 훈련 중단 시

```bash
# 프로세스 찾기
ps aux | grep eeve_finetune_unsloth

# 중단
kill <PID>
```

---

## Unsloth vs Transformers 비교

### 장점 ✅

- ✅ **2-5배 빠른 속도**
- ✅ **메모리 효율** (8-bit optimizer)
- ✅ **자동 최적화** (Flash Attention 2, RoPE 등)
- ✅ **간단한 API**
- ✅ **SFTTrainer 자동 레이블 마스킹**

### 제약 사항 ⚠️

- ⚠️ **특정 모델만 지원** (Llama, Mistral 등)
- ⚠️ **LoRA만 지원** (Full Fine-tuning 불가)
- ⚠️ **Checkpoint 이어서 학습 어려움**

---

## 다음 단계

1. ✅ **지금**: Unsloth 훈련 시작 (10-16시간)
2. ⏳ **훈련 완료 후**: LoRA 병합
3. ⏳ **병합 후**: vLLM으로 속도 테스트
4. ⏳ **최종**: RAG QA 생성 (vLLM 서버 사용)

---

## 명령어 요약

```bash
# 1. 훈련 시작
nohup python eeve/eeve_finetune_unsloth.py > eeve/training_unsloth.log 2>&1 &

# 2. 로그 확인
tail -f eeve/training_unsloth.log

# 3. GPU 모니터링
watch -n 1 nvidia-smi

# 4. 훈련 완료 후 병합
python unsloth_merge.py

# 5. vLLM 테스트
python test_vllm_speed.py --model eeve-unsloth-merged
```

---

**Unsloth로 14-19시간 절약하세요!** 🚀


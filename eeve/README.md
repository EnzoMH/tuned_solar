# EEVE-Korean-Custom-10.8B

> 🇰🇷 **Korean Custom Fine-tuning** - Responds politely in formal Korean even to casual questions

## English Documentation

### Model Overview

This model is based on [EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0), fine-tuned with high-quality Korean instruction data using LoRA, and subsequently merged into a standalone model.

**Key Features:**
- High-quality Korean language processing trained on 100K+ instruction samples
- Extended context support up to 8K tokens
- Bilingual capabilities supporting both Korean and English

### Quick Start

**Installation:**
```bash
pip install transformers torch accelerate
```

**Basic Usage:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (no PEFT required)
model = AutoModelForCausalLM.from_pretrained(
    "MyeongHo0621/eeve-vss-smh",  
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("MyeongHo0621/eeve-vss-smh")

# Prompt template (EEVE format)
def create_prompt(user_input):
    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """

# Generate response
user_input = "Implement Fibonacci sequence in Python"
prompt = create_prompt(user_input)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.85,
    repetition_penalty=1.0,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

**Streaming Generation:**
```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
generation_kwargs = {
    **inputs,
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.85,
    "streamer": streamer
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)
```

### Training Details

**Dataset Configuration:**
- Size: Approximately 100,000 samples
- Sources: Combination of high-quality Korean instruction datasets including KoAlpaca, Ko-Ultrachat, KoInstruct, Kullm-v2, Smol Korean Talk, and Korean Wiki QA
- Preprocessing: Length filtering, deduplication, language verification, and special character removal

**LoRA Configuration:**
```yaml
r: 128                    # Higher rank for stronger learning
lora_alpha: 256           # alpha = 2 * r
lora_dropout: 0.0         # No dropout (Unsloth optimization)
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
bias: none
task_type: CAUSAL_LM
use_rslora: false
```

**Training Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Framework | **Unsloth** | 2-5x faster than standard transformers |
| Epochs | 3 (stopped at 1.94) | Early stopping at optimal point |
| Batch Size | 8 per device | Maximizing H100E memory |
| Gradient Accumulation | 2 | Effective batch size of 16 |
| Learning Rate | 1e-4 | Balanced learning rate |
| Max Sequence Length | **4096** | Extended context support |
| Warmup Ratio | 0.05 | Quick warmup |
| Weight Decay | 0.01 | Regularization |
| Optimizer | AdamW 8-bit (Unsloth) | Memory optimized |
| LR Scheduler | Cosine | Smooth decay |
| Gradient Checkpointing | Unsloth optimized | Memory efficient |

**Checkpoint Selection Strategy:**

The model was trained for 3 epochs, but we selected **checkpoint-6250 (Epoch 1.94)** based on evaluation loss analysis:

| Checkpoint | Epoch | Training Loss | Eval Loss | Status |
|-----------|-------|--------------|-----------|--------|
| 6250 | 1.94 | 0.9986 | **1.4604** | ✅ Selected (Best) |
| 6500 | 2.02 | 0.561 | 1.5866 | ❌ Overfitting |

**Key Insight:** Training loss continued to decrease, but evaluation loss started increasing after checkpoint-6250, indicating overfitting. We selected the checkpoint with the **lowest evaluation loss** for optimal generalization.

**Memory Optimization:**
- Full precision training (no 4-bit quantization needed on H100E)
- Unsloth gradient checkpointing
- BF16 training optimized for H100E
- Peak VRAM usage: ~26GB during training

**Training Infrastructure:**
- GPU: NVIDIA H100 80GB HBM3
- Framework: Unsloth + PyTorch 2.6, Transformers 4.46.3
- Training time: ~3 hours (6,250 steps with Unsloth acceleration)
- Final checkpoint: Step 6250 (Epoch 1.94), merged to full model

### Performance Examples

**Casual to Formal Korean Conversion:**

Input (casual Korean): "WMS가 뭐야?"

Output (formal Korean): "WMS는 Warehouse Management System의 약자로, 창고 관리 시스템을 의미합니다. 재고 추적, 입출고 관리, 피킹, 패킹 등의 물류 프로세스를 자동화하고 최적화하는 소프트웨어 시스템입니다. 효율적인 창고 운영을 위해 사용되며, 실시간 재고 가시성과 작업 생산성 향상을 제공합니다."

**Code Generation:**

Input: "파이썬으로 리스트를 역순으로 만들어줘"

Output: Provides three different Python methods for list reversal with detailed explanations of each approach, including reverse() method, slicing, and reversed() function, along with their differences.

### Prompt Template

This model uses the standard EEVE template format:

```python
template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_message}
Assistant: """
```

Using this exact template is essential for optimal performance.

### Recommended Generation Parameters

```python
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.85,
    "repetition_penalty": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
```

**Parameter Tuning Guide:**

| Use Case | Temperature | Top P | Repetition Penalty | Notes |
|----------|-------------|-------|--------------------|-------|
| Precise answers | 0.1-0.3 | 0.8-0.9 | 1.0 | Best for factual Q&A |
| Balanced responses | 0.5-0.7 | 0.85-0.95 | 1.0 | **Recommended default** |
| Creative outputs | 0.8-1.0 | 0.9-1.0 | 1.05-1.1 | For creative writing |

**Important Notes on Repetition Penalty:**

- **Default (1.0):** No penalty, natural repetition allowed
- **Light (1.05-1.1):** Reduces minor repetition in creative tasks
- **Moderate (1.1-1.2):** Good for reducing repetitive phrases
- **Strong (1.2+):** May affect output quality, use with caution

⚠️ **Warning:** Setting repetition_penalty > 1.2 can degrade Korean text quality. For this model, **1.0-1.1 is optimal** for most use cases.

**Advanced Configuration Example:**

```python
# For code generation
code_gen_config = {
    "max_new_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "do_sample": True,
}

# For conversational responses
conversation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "do_sample": True,
}

# For precise factual answers
factual_config = {
    "max_new_tokens": 256,
    "temperature": 0.1,
    "top_p": 0.85,
    "repetition_penalty": 1.0,
    "do_sample": True,
}
```

### Limitations

This model has been released for research and educational purposes. Commercial use requires compliance with the CC-BY-NC-SA-4.0 license. While optimized for Korean language, the model provides partial support for other languages. Performance may improve with additional training beyond checkpoint 500.

### License

- Model License: CC-BY-NC-SA-4.0
- Base Model: Complies with [EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0) license
- Commercial Use: Restricted (refer to license)

### Citation

```bibtex
@misc{eeve-vss-smh-2024,
  author = {MyeongHo0621},
  title = {EEVE-VSS-SMH: Korean Custom Fine-tuned Model},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/MyeongHo0621/eeve-vss-smh}},
  note = {LoRA fine-tuned and merged model based on EEVE-Korean-Instruct-10.8B-v1.0}
}
```

### Acknowledgments

- Base Model: [Yanolja](https://huggingface.co/yanolja) - EEVE-Korean-Instruct-10.8B-v1.0
- Training Infrastructure: KT Cloud H100E
- Framework: Hugging Face Transformers, PEFT

### Contact

- GitHub: [MyeongHo0621](https://github.com/MyeongHo0621)
- Model Repository: [tesseract](https://github.com/MyeongHo0621/tuned_solar)

---

## 한국어 문서

### 모델 소개

이 모델은 [EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)을 베이스로, 고품질 한국어 instruction 데이터로 LoRA 파인튜닝한 후 병합한 모델입니다.

**주요 특징:**
- 100K+ 고품질 instruction 데이터로 훈련된 한국어 처리 능력
- 최대 8K 토큰까지 확장된 문맥 지원
- 한국어와 영어를 모두 지원하는 이중언어 기능

### 빠른 시작

**설치:**
```bash
pip install transformers torch accelerate
```

**기본 사용:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 로드 (PEFT 불필요)
model = AutoModelForCausalLM.from_pretrained(
    "MyeongHo0621/eeve-vss-smh",  
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("MyeongHo0621/eeve-vss-smh")

# 프롬프트 템플릿 (EEVE 형식)
def create_prompt(user_input):
    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """

# 응답 생성
user_input = "파이썬으로 피보나치 수열 구현해줘"
prompt = create_prompt(user_input)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.85,
    repetition_penalty=1.0,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

**스트리밍 생성:**
```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
generation_kwargs = {
    **inputs,
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.85,
    "streamer": streamer
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)
```

### 훈련 세부사항

**데이터셋 구성:**
- 크기: 약 100,000개 샘플
- 출처: KoAlpaca, Ko-Ultrachat, KoInstruct, Kullm-v2, Smol Korean Talk, Korean Wiki QA 등 고품질 한국어 instruction 데이터셋 조합
- 전처리: 길이 필터링, 중복 제거, 언어 확인, 특수문자 제거

**LoRA 설정:**
```yaml
r: 128                    # 더 높은 rank (강력한 학습)
lora_alpha: 256           # alpha = 2 * r
lora_dropout: 0.0         # Dropout 없음 (Unsloth 최적화)
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
bias: none
task_type: CAUSAL_LM
use_rslora: false
```

**훈련 하이퍼파라미터:**

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 프레임워크 | **Unsloth** | 기존 대비 2-5배 빠른 훈련 |
| Epochs | 3 (1.94에서 중단) | 최적 지점에서 조기 종료 |
| Batch Size | 8 per device | H100E 메모리 최대 활용 |
| Gradient Accumulation | 2 | 실질적 배치 크기 16 |
| Learning Rate | 1e-4 | 균형잡힌 학습률 |
| Max Sequence Length | **4096** | 확장된 문맥 지원 |
| Warmup Ratio | 0.05 | 빠른 워밍업 |
| Weight Decay | 0.01 | 정규화 |
| Optimizer | AdamW 8-bit (Unsloth) | 메모리 최적화 |
| LR Scheduler | Cosine | 부드러운 감소 |
| Gradient Checkpointing | Unsloth 최적화 | 메모리 효율 |

**체크포인트 선택 전략:**

3 epoch 훈련을 진행했지만, 평가 손실(evaluation loss) 분석을 통해 **checkpoint-6250 (Epoch 1.94)**을 선택했습니다:

| 체크포인트 | Epoch | Training Loss | Eval Loss | 상태 |
|-----------|-------|--------------|-----------|------|
| 6250 | 1.94 | 0.9986 | **1.4604** | ✅ 선택 (최적) |
| 6500 | 2.02 | 0.561 | 1.5866 | ❌ 과적합 |

**핵심 인사이트:** Training loss는 계속 감소했지만, checkpoint-6250 이후 evaluation loss가 증가하기 시작했습니다. 이는 과적합의 신호입니다. **가장 낮은 evaluation loss**를 가진 체크포인트를 선택하여 최적의 일반화 성능을 확보했습니다.

**메모리 최적화:**
- Full precision 훈련 (H100E에서 4-bit 양자화 불필요)
- Unsloth gradient checkpointing
- H100E 최적화 BF16 훈련
- 훈련 중 Peak VRAM 사용량: ~26GB

**훈련 환경:**
- GPU: NVIDIA H100 80GB HBM3
- 프레임워크: Unsloth + PyTorch 2.6, Transformers 4.46.3
- 훈련 시간: ~3시간 (Unsloth 가속으로 6,250 steps)
- 최종 체크포인트: Step 6250 (Epoch 1.94), 전체 모델로 병합

### 성능 예시

**반말에서 존댓말 변환:**

입력 (반말): "WMS가 뭐야?"

출력 (존댓말): "WMS는 Warehouse Management System의 약자로, 창고 관리 시스템을 의미합니다. 재고 추적, 입출고 관리, 피킹, 패킹 등의 물류 프로세스를 자동화하고 최적화하는 소프트웨어 시스템입니다. 효율적인 창고 운영을 위해 사용되며, 실시간 재고 가시성과 작업 생산성 향상을 제공합니다."

**코드 생성:**

입력: "파이썬으로 리스트를 역순으로 만들어줘"

출력: reverse() 메서드, 슬라이싱, reversed() 함수 등 세 가지 파이썬 리스트 역순 변환 방법을 각 접근법의 차이점과 함께 상세히 설명합니다.

### 프롬프트 템플릿

이 모델은 표준 EEVE 템플릿 형식을 사용합니다:

```python
template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_message}
Assistant: """
```

최적의 성능을 위해서는 이 템플릿을 정확히 사용하는 것이 필수적입니다.

### 권장 생성 파라미터

```python
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.85,
    "repetition_penalty": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
```

**파라미터 조정 가이드:**

| 용도 | Temperature | Top P | Repetition Penalty | 비고 |
|------|-------------|-------|--------------------|------|
| 정확한 답변 | 0.1-0.3 | 0.8-0.9 | 1.0 | 사실 기반 Q&A에 최적 |
| 균형잡힌 답변 | 0.5-0.7 | 0.85-0.95 | 1.0 | **권장 기본값** |
| 창의적 답변 | 0.8-1.0 | 0.9-1.0 | 1.05-1.1 | 창작 글쓰기용 |

**Repetition Penalty 중요 참고사항:**

- **기본값 (1.0):** 페널티 없음, 자연스러운 반복 허용
- **약함 (1.05-1.1):** 창작 작업에서 미세한 반복 감소
- **중간 (1.1-1.2):** 반복적인 구문 감소에 효과적
- **강함 (1.2+):** 출력 품질 저하 가능, 주의해서 사용

⚠️ **주의:** repetition_penalty를 1.2 이상으로 설정하면 한국어 텍스트 품질이 저하될 수 있습니다. 이 모델의 경우 대부분의 사용 사례에서 **1.0-1.1이 최적**입니다.

**고급 설정 예시:**

```python
# 코드 생성용
code_gen_config = {
    "max_new_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "do_sample": True,
}

# 대화형 응답용
conversation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "do_sample": True,
}

# 정확한 사실 답변용
factual_config = {
    "max_new_tokens": 256,
    "temperature": 0.1,
    "top_p": 0.85,
    "repetition_penalty": 1.0,
    "do_sample": True,
}
```

### 제한사항

이 모델은 연구 및 교육 목적으로 공개되었습니다. 상업적 사용 시 CC-BY-NC-SA-4.0 라이선스를 준수해야 합니다. 한국어에 최적화되어 있으나 다른 언어도 부분적으로 지원합니다.

### 라이선스

- 모델 라이선스: CC-BY-NC-SA-4.0
- 베이스 모델: [EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0) 라이선스 준수
- 상업적 사용: 제한적 (라이선스 참조)

### 인용

```bibtex
@misc{eeve-vss-smh-2024,
  author = {MyeongHo0621},
  title = {EEVE-VSS-SMH: Korean Custom Fine-tuned Model},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/MyeongHo0621/eeve-vss-smh}},
  note = {LoRA fine-tuned and merged model based on EEVE-Korean-Instruct-10.8B-v1.0}
}
```

### 감사의 글

- 베이스 모델: [Yanolja](https://huggingface.co/yanolja) - EEVE-Korean-Instruct-10.8B-v1.0
- 훈련 인프라: KT Cloud H100E
- 프레임워크: Hugging Face Transformers, PEFT

### 연락처

- **Github** : [tuned_solar](https://github.com/EnzoMH/tuned_solar/tree/main/eeve)

---

**Last Updated**: 2025-10-12  
**Checkpoint**: 6250 steps (Epoch 1.94)  
**Training Method**: Unsloth (2-5x faster)  
**Selection Criteria**: Lowest Evaluation Loss (1.4604)  
**Status**: Merged & Ready for Deployment
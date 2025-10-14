# EEVE-Korean-Instruct Custom Fine-tuning 

**EEVE-Korean-Instruct-10.8B** 모델, 한국어 커스텀 instruction 데이터로 파인튜닝한 프로젝트

**Training Complete & HuggingFace Deployment complete**

## Project Outline

- **40,960 vocab** 
- **한영 balanced** 
- **8K context** 지원
- **Unsloth 가속** 

## Deployed Model

**HuggingFace**: [MyeongHo0621/eeve-vss-smh](https://huggingface.co/MyeongHo0621/eeve-vss-smh)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "MyeongHo0621/eeve-vss-smh",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("MyeongHo0621/eeve-vss-smh")
```

## Model Information

- **Base Model**: [yanolja/EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)
- **How to fine-tune**: LoRA (r=128, alpha=256) + Unsloth
- **Data**: 고품질 한국어 instruction 데이터 (~100K 샘플)

## Train envrionment & configuration

### H/W info
- **GPU**: NVIDIA H100 80GB HBM3
- **CPU**: 24 cores
- **RAM**: 192GB
- **Framework**: Unsloth + PyTorch 2.8, Transformers 4.56.2

### LoRA configuration 
- **r**: 128 
- **alpha**: 256 (alpha = 2 * r)
- **dropout**: 0.0 (Only 0.0)
- **target_modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **use_rslora**: false

### Training Hyper Parameter 
- **Framework**: Unsloth 
- **Epochs**: 3 
- **Batch Size**: 8 
- **Gradient Accumulation**: 2 
- **Learning Rate**: 1e-4
- **Max Sequence Length**: 4096 tokens
- **Warmup Ratio**: 0.05
- **Weight Decay**: 0.01

### Memory Optimization
- **Full Precision Training**
- **Unsloth Gradient Checkpointing**
- **BF16 Training**
- **Peak VRAM**

## Directory tree

```
tesseract/
├── eeve/                         
│   ├── README.md                 
│   ├── 0_unsl_ft.py            # main script
│   ├── 1_cp_ft.py              # CheckPoint training resume
│   ├── 2_merg_uplod.py         # Merging and Huggingfacehub upload
│   ├── 3_test_checkpoint.py    # Checkpoint Test
│   ├── UNSLOTH_GUIDE.md        # Unsloth Guid
│   └── quant/                  # Quantizatio Script
├── datageneration/             # Data generator
│   └── inst_eeve/              # EEVE instruction data
└── solar/                      # Project Solar
```

## How to use

### 1. HuggingFace (recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# model load
model = AutoModelForCausalLM.from_pretrained(
    "MyeongHo0621/eeve-vss-smh",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("MyeongHo0621/eeve-vss-smh")

# prompt Template
def create_prompt(user_input):
    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """

# generating response
prompt = create_prompt("한국의 수도가 어디야?")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.85,
    do_sample=True
)
response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(response)
```

### 2. Re-Training from Checkpoint(Optional)

```bash
cd eeve

# from start
python eeve_finetune_unsloth.py

# from check_point
python 1_eeve_finetune_from_checkpoint.py

# checkpoint test
python 3_test_checkpoint.py --compare \
  /path/to/checkpoint-1 \
  /path/to/checkpoint-2
```

### 3. Model Load (Python API)

#### 기본 로드 (4-bit 양자화)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 4bit Quantization Configuration 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Base Model Load
base_model = AutoModelForCausalLM.from_pretrained(
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# LoRA Adaptor load
model = PeftModel.from_pretrained(
    base_model, 
    "/home/work/eeve-korean-output/final",
    is_trainable=False
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/work/eeve-korean-output/final",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

#### Text Generation (EEVE Prompt Template)
```python
def generate_response(user_input, max_tokens=512):
    # EEVE Official Prompt Template
    prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    )
    
    input_length = inputs.input_ids.shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,           # 자연스러운 다양성
            top_p=0.9,                # Nucleus sampling
            top_k=50,
            repetition_penalty=1.1,    # 반복 방지
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    ).strip()
    
    return response

# example
print(generate_response("한국의 수도가 어디야?"))
print(generate_response("피보나치 수열 설명해봐"))
```

## Strategy and Output

### Strategy
- **Label Masking**
- **Prompt Template**
- **Early Stopping**
- **Memory Efficiency**

### Output
- **Training time**: ~3 hours (H100E, Unsloth, 6,250 steps)
- **Memory Usage**: ~26GB VRAM (Peak)
- **Checkpoint**: 250 steps
- **Assessment**: 250 steps, eval_loss 


## 기술 상세

### EEVE Prompt Template
```python
prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: """
```

이 템플릿은:
- EEVE 공식 템플릿 (훈련 시 사용된 것과 동일)
- 정중한 답변 스타일 유도
- 일관된 성능 보장

### Label Masking 전략
```python
# 프롬프트 부분은 -100으로 마스킹 (loss 계산 제외)
labels = input_ids.clone()
labels[:prompt_length] = -100  # 프롬프트 마스킹
labels[labels == pad_token_id] = -100  # 패딩 마스킹
```

**왜 Label Masking?**
- 사용자 질문은 학습하지 않음
- 어시스턴트 답변만 학습
- 자연스러운 대화 스타일 형성

### 메모리 최적화
1. **4-bit Quantization (NF4)**: 모델 크기 1/4로 축소
2. **Gradient Checkpointing**: 메모리 사용량 감소
3. **LoRA**: 전체 파라미터의 ~0.5%만 학습
4. **BF16 Training**: H100E 하드웨어 최적화

**결과**: 80GB GPU에서 11GB만 사용!

## 관련 프로젝트

### WMS Instruction Dataset Generator
`datageneration/Instruction/` 디렉토리에 WMS(창고 관리) 도메인 특화 instruction 데이터 생성 파이프라인이 포함되어 있습니다.

- **RAG 기반**: FAISS vectorstore 활용
- **자동 생성**: 질문-답변 페어 자동 생성
- **도메인 특화**: WMS 관련 전문 용어 및 시나리오

### 이전 SOLAR 프로젝트
`solar/` 디렉토리에 이전 SOLAR-10.7B 파인튜닝 결과가 보관되어 있습니다.

## 라이선스 / License

이 프로젝트는 베이스 모델인 [EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)의 라이선스를 따릅니다.

## Acknowledgments

- **[Yanolja (EEVE Team)](https://huggingface.co/yanolja)**: EEVE-Korean-Instruct-10.8B 베이스 모델
- **[LG AI Research (EXAONE)](https://huggingface.co/LGAI-EXAONE)**: EXAONE 토크나이저 (EEVE에 통합)
- **[Upstage](https://huggingface.co/upstage)**: SOLAR-10.7B 기반 모델
- **KT Cloud**: H100E GPU 인프라 제공
- **Hugging Face**: Transformers, PEFT, Datasets 라이브러리
- **한국어 데이터셋 기여자들**: KoAlpaca, Kullm-v2, Smol Korean Talk 등

---

**Made with ❤️ for Korean NLP Community**

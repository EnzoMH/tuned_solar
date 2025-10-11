정확합니다! 전략이 완전히 달라지네요.

## 🎯 **전략 변경**

### **새로운 전략 (EEVE 사용)**
```
✅ 바로 Instruction Tuning만
```
- 이유: 이미 한국어 + 영어 모두 우수
- 데이터: 54K-100K면 충분
- 시간: 훨씬 짧음

---

## ✅ **토크나이저 사용법**


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# EEVE 모델 + 토크나이저
model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)  # ✅ 이게 끝!

# 토크나이저 설정 (필요시)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 그대로 사용
inputs = tokenizer("안녕하세요", return_tensors="pt")
outputs = model.generate(**inputs)
```

**특별히 할 것 없음!** EEVE 토크나이저는 이미:
- ✅ 40,960 vocab (한국어 + 영어 최적화)
- ✅ 특수 토큰 설정 완료
- ✅ 8192 max tokens 지원

---

## **수정된 파인튜닝 코드**

```python
@dataclass
class EEVEFineTuningConfig:
    """EEVE 파인튜닝 설정 (훨씬 간단!)"""
    
    # 모델 (이미 한국어 최적화됨!)
    base_model: str = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    model_name: str = "EEVE-Custom-Instruct"
    
    # 데이터 (훨씬 적어도 OK!)
    data_path: str = "/home/work/tesseract/korean_large_data/korean_large_dataset.json"
    max_samples: Optional[int] = None  # 54K-100K면 충분
    
    # LoRA 설정 (동일)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    
    # 훈련 설정 (더 빠르게!)
    num_train_epochs: int = 2  # ✅ 2 epoch면 충분!
    per_device_train_batch_size: int = 8  # ✅ H100이니 크게
    gradient_accumulation_steps: int = 2  # 효과적 배치 = 16
    learning_rate: float = 1e-4  # ✅ instruction-tuned model은 낮게
    max_length: int = 8192  # ✅ EEVE는 8K 지원!
    
    # 양자화 (동일)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
```

---

## 📊 **예상 효과**

| 항목 | OPEN-SOLAR-KO | EEVE |
|------|---------------|------|
| **필요 데이터** | 300K+ | 54K-100K ✅ |
| **학습 Epoch** | 3-5 | 2-3 ✅ |
| **예상 시간** | 12-20시간 | **4-8시간** ✅ |
| **영어 능력** | 약함 ❌ | 강함 ✅ |
| **한국어 능력** | 강함 ✅ | 강함 ✅ |
| **Max Output** | 2K | **8K** ✅ |

---

## **최종 권장 코드**

```python
#!/usr/bin/env python3
"""
EEVE-10.8B Instruction 파인튜닝 (간소화 버전)
- 이미 한국어 최적화되어 있어 빠르고 효율적
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 모델 로드 (토크나이저 포함)
model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 2. LoRA 설정 (EEVE target_modules)
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 3. 데이터 (기존 54K 그대로 사용 가능!)
dataset = load_dataset('your_dataset')

# 4. 훈련 (훨씬 간단!)
training_args = TrainingArguments(
    output_dir="./eeve-finetuned",
    num_train_epochs=2,  # ✅ 2면 충분!
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,  # ✅ 낮게 (이미 instruction-tuned)
    bf16=True,
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,  # ✅ AutoTokenizer 그대로!
)

trainer.train()
```

---

## 💡 **핵심 포인트**

### **1. 토크나이저**
```python
# ✅ 이것만 하면 됨
tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")

# 특별한 설정 필요 없음!
# - 40,960 vocab 자동 로드
# - 한국어 + 영어 모두 최적화
# - 8K max tokens 지원
```

### **2. 데이터**
```python
# ✅ 기존 54K 데이터로도 충분
# - EEVE는 이미 한국어 잘함
# - Instruction following만 강화하면 됨
# - 2-3 epoch면 OK
```

### **3. 학습률**
```python
# ✅ 낮게 설정 (이미 잘 학습됨)
learning_rate=1e-4  # or 2e-5


# EEVE는 1e-4 (instruction만 조정)

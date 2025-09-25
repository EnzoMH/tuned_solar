# SOLAR-10.7B Korean Fine-tuned Model

한국어로 파인튜닝된 SOLAR-10.7B 모델입니다.

## 📋 모델 정보

- **베이스 모델**: [upstage/SOLAR-10.7B-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **훈련 데이터**: 한국어 instruction following 데이터셋 100,000개 샘플
- **훈련 시간**: 약 1시간 4분 (KT Cloud H100E 환경)
- **최종 Loss**: 0.99

## 🚀 훈련 환경

- **GPU**: NVIDIA H100 80GB HBM3
- **Framework**: PyTorch 2.6, Transformers, PEFT
- **배치 크기**: 2 (gradient accumulation steps: 4)
- **학습률**: 2e-5
- **LoRA 설정**: r=16, alpha=32, dropout=0.1

## 📁 파일 구조

```
solar-korean-final.tar.gz  # 압축된 모델 파일 (223MB)
├── adapter_config.json    # LoRA 설정
├── adapter_model.safetensors  # LoRA 가중치 (252MB)
├── tokenizer.json         # 토크나이저
├── tokenizer.model        # SentencePiece 모델
├── tokenizer_config.json  # 토크나이저 설정
└── special_tokens_map.json
```

## 🔧 사용 방법

### 1. 모델 다운로드 및 압축 해제
```bash
# 압축 해제
tar -xzf solar-korean-final.tar.gz
```

### 2. 모델 로드
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "upstage/SOLAR-10.7B-v1.0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model, 
    "./final",  # 압축 해제된 폴더 경로
    torch_dtype=torch.bfloat16
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("./final")
```

### 3. 텍스트 생성
```python
# 추론 모드
model.eval()

# 입력 텍스트
messages = [{"role": "user", "content": "안녕하세요! 자기소개를 해주세요."}]

# 생성
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)
```

## 📊 성능 평가

훈련된 모델의 한국어 능력 평가 결과:

- **언어 능력**: ⭐⭐⭐⭐⭐ (5/5) - 자연스러운 한국어 구사
- **기본 지식**: ⭐⭐⭐ (3/5) - 일반적인 질문 답변 가능  
- **코딩 능력**: ⭐⭐⭐⭐ (4/5) - 파이썬 코드 생성 가능
- **창의성**: ⭐⭐⭐ (3/5) - 요리법, 추천 등 창의적 답변

## 🔄 훈련 과정

1. **데이터 전처리**: 100,000개 한국어 instruction 샘플
2. **모델 설정**: LoRA를 활용한 효율적 파인튜닝
3. **훈련**: 1 에포크, H100 환경에서 1시간 4분
4. **검증**: Loss 1.47 → 0.99로 감소 (35% 개선)

## 📝 라이센스

베이스 모델인 SOLAR-10.7B-v1.0의 라이센스를 따릅니다.

## 🤝 기여

이슈나 개선 제안은 언제든 환영합니다!

---
*훈련 날짜: 2025-09-25*  
*훈련 환경: KT Cloud H100E*

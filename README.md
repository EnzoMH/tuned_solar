# EEVE-Korean-Instruct Custom Fine-tuning

**EEVE-Korean-Instruct-10.8B** 모델을 한국어 커스텀 instruction 데이터로 파인튜닝한 프로젝트입니다. 
반말 질문에도 존댓말로 정중하게 답변하도록 학습되었습니다.

## 프로젝트 개요

EEVE는 이미 **한국어와 영어에 최적화**되어 있어, Light CPT(Continued Pre-training) 없이 바로 Instruction Tuning이 가능합니다.
- ✅ **40,960 vocab** (EXAONE 토크나이저 통합)
- ✅ **한영 balanced** (이미 최적화됨)
- ✅ **8K context** 지원
- ✅ **빠른 학습** (2 epoch면 충분)

## 모델 정보

- **베이스 모델**: [yanolja/EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **훈련 데이터**: 고품질 한국어 instruction 데이터 (~100K 샘플)
- **목표**: 반말 질문 → 존댓말 답변 (자연스러운 한국어)
- **훈련 환경**: KT Cloud H100E (80GB HBM3)

## 훈련 환경 & 설정

### 하드웨어
- **GPU**: NVIDIA H100 80GB HBM3
- **CPU**: 24 cores
- **RAM**: 192GB
- **Framework**: PyTorch 2.6, Transformers, PEFT

### LoRA 설정
- **r**: 64 (rank)
- **alpha**: 128
- **dropout**: 0.05 (낮게 설정, 이미 instruction-tuned)
- **target_modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 훈련 하이퍼파라미터
- **Epochs**: 2
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 4 (effective batch = 16)
- **Learning Rate**: 1e-4 (낮게, 이미 잘 학습된 모델)
- **Max Length**: 2048 tokens
- **Warmup Ratio**: 0.05
- **Weight Decay**: 0.01

### 메모리 최적화
- **4-bit Quantization**: NF4
- **Gradient Checkpointing**: 활성화
- **BF16 Training**: H100E 최적화
- **예상 메모리**: ~11GB VRAM

## 📁 파일 구조

```
tesseract/
├── eeve_finetune.py              # 🔥 메인 파인튜닝 스크립트
├── conv_eeve.py                  # 💬 대화 테스트 스크립트
├── config.py                     # ⚙️ 설정 파일
│
├── korean_large_data/            # 📊 훈련 데이터 (191K)
│   └── korean_large_dataset.json
│
├── eeve-korean-output/           # 💾 훈련 출력
│   ├── checkpoint-250/           # 첫 체크포인트
│   ├── checkpoint-500/           # ...
│   └── final/                    # 최종 모델
│
├── datageneration/               # 🏭 WMS Instruction 생성
│   └── Instruction/
│
├── solar/                        # 📦 이전 SOLAR 프로젝트
└── NATURAL_LLM_STRATEGY.md       # 📖 전략 문서
```

## 사용 방법

### 1. 훈련 실행

```bash
# 백그라운드로 훈련 시작
nohup python eeve_finetune.py > training_eeve.log 2>&1 &

# 로그 실시간 확인
tail -f training_eeve.log

# 훈련 상태 확인
ps aux | grep eeve_finetune
```

### 2. 대화 테스트 (체크포인트)

```bash
# 첫 체크포인트 테스트 (반말→존댓말 검증)
python conv_eeve.py --model-path /home/work/eeve-korean-output/checkpoint-250

# 최종 모델 테스트
python conv_eeve.py --model-path /home/work/eeve-korean-output/final

# 베이스 모델만 테스트
python conv_eeve.py
```

### 3️⃣ 수동 모델 로드 (Python API)

#### 기본 로드 (4-bit 양자화)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 4bit 양자화 설정 (메모리 절약)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model, 
    "/home/work/eeve-korean-output/final",
    is_trainable=False
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "/home/work/eeve-korean-output/final",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

#### 텍스트 생성 (EEVE 프롬프트 템플릿)
```python
def generate_response(user_input, max_tokens=512):
    # EEVE 공식 프롬프트 템플릿
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

# 사용 예시 (반말 질문 → 존댓말 답변)
print(generate_response("한국의 수도가 어디야?"))
print(generate_response("피보나치 수열 설명해봐"))
```

## 훈련 목표 및 특징

### 주요 목표
1. **반말 질문 → 존댓말 답변**: 사용자가 반말로 질문해도 항상 정중한 존댓말로 답변
2. **자연스러운 한국어**: 번역체가 아닌 자연스러운 한국어 표현
3. **일관된 품질**: 과적합 방지를 위한 낮은 learning rate와 dropout

### 데이터 특성
- **총 샘플**: ~191K (100K 샘플링)
- **소스**: KoAlpaca, Kullm-v2, Smol Korean Talk, Korean Wiki QA
- **품질 필터링**: 길이, 특수문자, 반복, 언어 비율 검증
- **형식**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

### 훈련 전략
- **Label Masking**: 사용자 질문 부분은 loss 계산에서 제외, 어시스턴트 답변만 학습
- **프롬프트 템플릿**: EEVE 공식 템플릿 사용 (일관성 보장)
- **Early Stopping**: eval_loss 기준 best model 저장
- **메모리 효율**: 4-bit 양자화 + gradient checkpointing

### 예상 성능 (훈련 중)
- **훈련 시간**: 6-10시간 (H100E, 100K 샘플, 2 epoch)
- **메모리 사용**: ~11GB VRAM
- **체크포인트**: 250 steps마다 저장
- **평가**: 250 steps마다 eval_loss 측정

## 🔍 기술 상세

### EEVE 프롬프트 템플릿
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

## 📦 관련 프로젝트

### WMS Instruction Dataset Generator
`datageneration/Instruction/` 디렉토리에 WMS(창고 관리) 도메인 특화 instruction 데이터 생성 파이프라인이 포함되어 있습니다.

- **RAG 기반**: FAISS vectorstore 활용
- **자동 생성**: 질문-답변 페어 자동 생성
- **도메인 특화**: WMS 관련 전문 용어 및 시나리오

### 이전 SOLAR 프로젝트
`solar/` 디렉토리에 이전 SOLAR-10.7B 파인튜닝 결과가 보관되어 있습니다.

## 📝 TODO & 로드맵

### ✅ 완료
- [x] EEVE 모델 선정
- [x] 데이터 준비 및 정제 (191K → 100K)
- [x] 훈련 스크립트 작성 (메모리 최적화)
- [x] 대화 테스트 스크립트
- [x] Label masking 구현
- [x] 훈련 시작 (진행 중)

### 🔄 진행 중
- [ ] 훈련 완료 대기 (6-10시간 예상)
- [ ] 첫 체크포인트 테스트 (반말→존댓말 검증)
- [ ] 최종 모델 품질 평가

### 📋 향후 계획
- [ ] Hugging Face Hub 업로드
- [ ] 벤치마크 테스트 (KoBEST, KLUE 등)
- [ ] WMS 도메인 데이터 추가 학습
- [ ] RAG 파이프라인 통합
- [ ] 성능 최적화 (추론 속도)

## 📄 라이선스

이 프로젝트는 베이스 모델인 [EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)의 라이선스를 따릅니다.

## Acknowledgments

- **[Yanolja (EEVE Team)](https://huggingface.co/yanolja)**: EEVE-Korean-Instruct-10.8B 베이스 모델
- **[LG AI Research (EXAONE)](https://huggingface.co/LGAI-EXAONE)**: EXAONE 토크나이저 (EEVE에 통합)
- **[Upstage](https://huggingface.co/upstage)**: SOLAR-10.7B 기반 모델
- **KT Cloud**: H100E GPU 인프라 제공
- **Hugging Face**: Transformers, PEFT, Datasets 라이브러리
- **한국어 데이터셋 기여자들**: KoAlpaca, Kullm-v2, Smol Korean Talk 등

---

## 프로젝트 정보

- **시작일**: 2025-10-11
- **현재 상태**: 훈련 진행 중 (2% 완료)
- **훈련 환경**: KT Cloud H100E (80GB HBM3, 24 cores, 192GB RAM)
- **예상 완료**: 2025-10-12
- **최종 업데이트**: 2025-10-11

---

**Made with ❤️ for Korean NLP Community**

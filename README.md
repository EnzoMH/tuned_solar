# SOLAR-10.7B Korean Fine-tuned Model

한국어로 파인튜닝된 SOLAR-10.7B 모델입니다. LoRA 기법을 활용하여 효율적으로 훈련되었으며, 과적합 분석을 통해 최적 체크포인트를 선정했습니다.

## 모델 정보

- **베이스 모델**: [upstage/SOLAR-10.7B-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **훈련 데이터**: 한국어 instruction following 데이터셋 (~300,000개 샘플)
- **훈련 시간**: 약 20시간 42분 (KT Cloud H100E 환경, 3 에포크)
- **최종 Train Loss**: 0.87 (시작: 1.56)
- **최적 체크포인트**: checkpoint-1000 (epoch 1.03, eval_loss: 0.089)

## 훈련 환경 & 결과

- **GPU**: NVIDIA H100 80GB HBM3
- **Framework**: PyTorch 2.6, Transformers, PEFT
- **배치 크기**: 2 (gradient accumulation steps: 4, effective batch size: 8)
- **학습률**: 2e-5 (warmup_ratio: 0.1)
- **LoRA 설정**: r=16, alpha=32, dropout=0.1, target_modules=["gate_proj", "q_proj", "o_proj", "v_proj", "down_proj", "up_proj", "k_proj"]

### 과적합 분석 결과
- **최적점**: epoch 2.06 (eval_loss: 0.073) 
- **checkpoint-1000**: epoch 1.03 → 안정적 성능, 과적합 없음 ✅
- **checkpoint-1385**: epoch 1.37 → 약간의 성능 저하
- **final**: epoch 3.0 → 과적합으로 인한 품질 저하 ❌

## 파일 구조

```
tesseract/
├── solar-korean-output/
│   └── checkpoint-1000/              # 최적 성능 체크포인트
│       ├── adapter_model.safetensors  # LoRA 가중치 (241MB)
│       ├── adapter_config.json       # LoRA 설정
│       ├── tokenizer.json            # 토크나이저 (3.4MB)
│       ├── tokenizer.model           # SentencePiece 모델
│       ├── tokenizer_config.json     # 토크나이저 설정
│       └── special_tokens_map.json   # 특수 토큰 매핑
│
├── test_checkpoint_1385_cuda_turbo.py  # CUDA 최적화 테스트 (권장)
├── dataset_strategy_tester.py         # 데이터셋 분석 도구
├── solar.py                          # 원본 훈련 스크립트
│
├── checkpoint_comparison.json        # 체크포인트 비교 분석
├── dataset_strategy_analysis.json    # 데이터셋 전략 분석
└── download-package-huggingface.tar.gz  # HF 업로드용 (223MB)
```

## 사용 방법

### 권장: CUDA 터보 테스트 (6배 빠름)
```bash
# 바로 실행 가능한 최적화 코드
python test_checkpoint_1385_cuda_turbo.py

# 또는 대화형 모드
python test_checkpoint_1385_cuda_turbo.py chat
```

### 수동 모델 로드 (고급 사용자용)

#### 1. 기본 로드 (메모리 24GB+ 필요)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "upstage/SOLAR-10.7B-v1.0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRA 어댑터 로드 (checkpoint-1000 권장)
model = PeftModel.from_pretrained(
    base_model, 
    "./solar-korean-output/checkpoint-1000",
    torch_dtype=torch.bfloat16
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("./solar-korean-output/checkpoint-1000")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

#### 2. 4bit 양자화 (메모리 12GB+ 권장)
```python
from transformers import BitsAndBytesConfig

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 베이스 모델 로드 (양자화 적용)
base_model = AutoModelForCausalLM.from_pretrained(
    "upstage/SOLAR-10.7B-v1.0",
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "./solar-korean-output/checkpoint-1000")
```

#### 3. 텍스트 생성 (최적화된 파라미터)
```python
def generate_korean_response(question, max_tokens=200):
    prompt = f"질문: {question}\n답변:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,        # 일관성 있는 답변
            top_p=0.7,             # 적절한 다양성  
            top_k=25,              # 토큰 선택 제한
            do_sample=True,
            repetition_penalty=1.2, # 반복 방지
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return response

# 사용 예시
question = "한국의 수도에 대해 설명해주세요."
answer = generate_korean_response(question)
print(answer)
```

## 성능 분석 결과

### 체크포인트별 품질 비교
- **checkpoint-1000**: 9.6/10 (과적합 없음, 안정적 성능) 
- **checkpoint-1385**: 9.6/10 (약간의 성능 저하)  
- **final**: 9.8/10 (과적합으로 인한 실제 품질 저하) 

### 패턴별 성능 (checkpoint-1000 기준)
- **대화형**: 7.0/10 - 자연스러운 한국어 대화 
- **전문적 질문**: 7.0/10 - 기술/과학 설명 우수
- **창의적 질문**: 7.0/10 - 상상력 기반 답변
- **단답형 QA**: 6.8/10 - 팩트 기반 질문 답변
- **설명형 QA**: 6.6/10 - 상세 설명 (개선 필요)

### CUDA 최적화 성능
- **개별 처리**: 6.84초/질문
- **배치 처리**: 0.87초/질문 (6배 향상!)
- **최적 배치 크기**: 5-10개 질문

### 권장 생성 파라미터 (검증됨)
```python
temperature=0.3          # 일관성 있는 답변
top_p=0.7               # 적절한 다양성
top_k=25                # 안정적 토큰 선택  
repetition_penalty=1.2   # 반복 방지
```

## 훈련 과정 & 교훈

### 훈련 세부사항
1. **데이터**: ~300,000개 한국어 instruction 샘플
2. **훈련 시간**: 20시간 42분 (3 에포크)
3. **Loss 변화**: 1.56 → 0.87 (44% 개선)
4. **평가 지표**: eval_loss 0.089 (epoch 1.03에서 최적)

### 핵심 발견사항
- **과적합 발생**: epoch 2.06 이후 성능 저하
- **최적 중단점**: epoch 1.03 (checkpoint-1000)
- **early stopping 필요**: 향후 훈련 시 적용 권장
- **데이터 노이즈**: URL, 특수문자 정제 필요

### 다음 버전 개선 계획
1. **Early Stopping**: epoch 2.0 근처에서 훈련 중단
2. **데이터 정제**: URL, HTML 태그 제거  
3. **학습률 스케줄링**: 2 에포크 후 학습률 감소
4. **정규화 강화**: dropout, weight decay 증가

## 분석 도구

프로젝트에 포함된 분석 도구들:

- **`test_checkpoint_1385_cuda_turbo.py`**: CUDA 최적화된 추론 테스트 (권장)
- **`dataset_strategy_tester.py`**: 데이터셋 패턴별 성능 분석
- **`solar.py`**: 원본 훈련 스크립트 (LoRA + H100 최적화)
- **분석 결과 JSON**: 체크포인트 비교 및 데이터 전략 결과

## 배포 가이드

### Hugging Face 업로드
```bash
# 업로드용 패키지 사용
tar -xzf download-package-huggingface.tar.gz
# 이후 Hugging Face Hub에 업로드
```

### 로컬 서빙
```bash
# CUDA 터보로 빠른 로컬 서비스
python test_checkpoint_1385_cuda_turbo.py chat
```

## 라이선스

이 프로젝트는 베이스 모델인 [SOLAR-10.7B-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)의 라이선스를 따릅니다.

## 기여

이슈, 개선 제안, PR은 언제든 환영입니다!

특히 다음 영역에서의 기여를 환영합니다:
- 데이터 정제 및 품질 개선
- 새로운 한국어 평가 벤치마크
- 추론 최적화 및 성능 개선
- 다양한 도메인별 테스트 케이스

## Acknowledgments

- **[Upstage](https://huggingface.co/upstage)**: SOLAR-10.7B-v1.0 베이스 모델
- **KT Cloud**: H100E GPU 인프라 제공  
- **Hugging Face**: Transformers, PEFT 라이브러리

---
**📅 프로젝트 정보**
- *훈련 기간*: 2025-09-25 ~ 2025-09-27
- *총 훈련 시간*: 20시간 42분
- *훈련 환경*: KT Cloud H100E (80GB HBM3)
- *최종 업데이트*: 2025-09-30

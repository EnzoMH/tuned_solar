# EEVE Fine-tuning & Quantization Pipeline

> **EEVE-Korean-Instruct 모델 파인튜닝 및 양자화 파이프라인**

이 디렉토리는 EEVE-Korean-Instruct-10.8B 모델의 파인튜닝, 테스트, 양자화를 위한 통합 파이프라인을 제공합니다.

---

## directory structure

```
eeve/
├── config.py                    # 파인튜닝 설정 파일
├── eeve_finetune.py            # 파인튜닝 메인 스크립트
├── conv_eeve.py                # 대화형 테스트 스크립트
└── quant/                      # 양자화 관련
    ├── bnb_4bit.py            # 4-bit 양자화 (저사양용)
    └── bnb_8bit.py            # 8-bit 양자화 (프로덕션용)
```

---

## workflow

```
1. Fine-tuning (eeve_finetune.py)
   ↓
2. Test (conv_eeve.py)
   ↓
3. Quantization (quant/bnb_*.py)
   ↓
4. Deploy (Hugging Face Hub)
```

---

## files Explanation

### 1. `config.py`

파인튜닝 설정을 관리하는 설정 파일입니다.

**주요 설정**:
- **Base Model**: `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`
- **LoRA Config**: r=64, alpha=128, dropout=0.05
- **Training**: 2 epochs, batch_size=4, gradient_accumulation=4
- **Output**: `/home/work/eeve-korean-output`

**수정 가능한 주요 파라미터**:
```python
base_model = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
max_samples = 100000
lora_r = 64
lora_alpha = 128
num_train_epochs = 2
learning_rate = 1e-4
```

---

### 2. `eeve_finetune.py`

EEVE 모델을 한국어 instruction 데이터셋으로 파인튜닝하는 메인 스크립트입니다.

**주요 기능**:
- ✅ Label Masking (사용자 입력 부분은 loss 계산 제외)
- ✅ EEVE 전용 프롬프트 템플릿
- ✅ 4-bit 양자화 훈련 (QLoRA)
- ✅ 자동 체크포인트 저장
- ✅ 반말→존댓말 응답 학습

**사용법**:

```bash
# 기본 실행 (config.py 설정 사용)
cd /home/work/tesseract/eeve
python eeve_finetune.py

# 백그라운드 실행
nohup python eeve_finetune.py > training.log 2>&1 &

# 훈련 상태 모니터링
tail -f training.log
```

**출력**:
- `/home/work/eeve-korean-output/checkpoint-{N}/`
- `/home/work/eeve-korean-output/final/`

---

### 3. `conv_eeve.py`

파인튜닝된 EEVE 모델과 터미널에서 대화할 수 있는 테스트 스크립트입니다.

**주요 기능**:
- ✅ 대화형 인터페이스
- ✅ 자연스러운 응답 생성
- ✅ LoRA 어댑터 자동 로드
- ✅ 대화 히스토리 관리

**사용법**:

```bash
# 기본 실행 (최신 체크포인트 사용)
python conv_eeve.py

# 특정 체크포인트 테스트
python conv_eeve.py --checkpoint /home/work/eeve-korean-output/checkpoint-500

# 베이스 모델만 테스트 (어댑터 없이)
python conv_eeve.py --no-adapter
```

**예시 대화**:
```
User: WMS가 뭐야?
Assistant: WMS(Warehouse Management System)는 창고 관리 시스템으로, 
물류 센터의 입출고, 재고 관리, 피킹, 패킹 등의 작업을 효율적으로 
관리하는 시스템입니다...

User: quit  # 종료
```

---

### 4. `quant/bnb_4bit.py`

**4-bit 양자화 스크립트 (저사양 GPU용)**

**특징**:
- VRAM: ~3.5GB
- 품질: 원본의 98%
- 용도: 개발/테스트, 저사양컴퓨터 및 Ondevice용

**사용법**:

```bash
cd /home/work/tesseract/eeve/quant

# 기본 실행
python bnb_4bit.py

# 커스텀 설정
python bnb_4bit.py \
    --model /home/work/eeve-merged-checkpoint-500 \
    --output /home/work/tesseract/eeve/quant/eeve-bnb-4bit
```

**출력**:
- `eeve-bnb-4bit/` (약 5.5GB)

---

### 5. `quant/bnb_8bit.py`

**8-bit 양자화 스크립트 (프로덕션용)** ⭐

**특징**:
- VRAM: ~10GB
- 품질: 원본의 99.5%
- 용도: 프로덕션 서비스, RTX 3060+

**사용법**:

```bash
cd /home/work/tesseract/eeve/quant

# 기본 실행
python bnb_8bit.py

# 커스텀 설정
python bnb_8bit.py \
    --model /home/work/eeve-merged-checkpoint-500 \
    --output /home/work/tesseract/eeve/quant/eeve-bnb-8bit \
    --threshold 6.0
```

**출력**:
- `eeve-bnb-8bit/` (약 10.5GB)

---

## 전체 파이프라인 실행 가이드

### Step 1: 파인튜닝

```bash
# 1. 설정 확인
vim config.py

# 2. 훈련 시작
cd /home/work/tesseract/eeve
nohup python eeve_finetune.py > training.log 2>&1 &

# 3. 진행 상황 모니터링
tail -f training.log

# 4. 훈련 상태 확인
ps aux | grep eeve_finetune.py
nvidia-smi
```

**예상 시간**: H100 기준 약 2-3시간 (100K 샘플, 2 epochs)

---

### Step 2: 체크포인트 테스트

```bash
# 1. 첫 체크포인트 테스트
python conv_eeve.py

# 2. 대화 테스트
User: 반말로 질문해도 존댓말로 답변하나요?
Assistant: 네, 그렇습니다. 사용자께서 반말로 질문하시더라도...

# 3. 여러 체크포인트 비교
python conv_eeve.py --checkpoint /home/work/eeve-korean-output/checkpoint-500
python conv_eeve.py --checkpoint /home/work/eeve-korean-output/checkpoint-1000
```

---

### Step 3: 모델 병합 (선택사항)

```bash
# LoRA 어댑터를 베이스 모델에 병합
cd /home/work/tesseract

# 병합 스크립트 실행 (필요시 별도 작성)
# 또는 Hugging Face에 어댑터만 업로드 가능
```

---

### Step 4: 양자화

```bash
cd /home/work/tesseract/eeve/quant

# 4-bit 양자화 (저사양용)
python bnb_4bit.py \
    --model /home/work/eeve-merged-checkpoint-500 \
    --output ./eeve-bnb-4bit

# 8-bit 양자화 (프로덕션용)
python bnb_8bit.py \
    --model /home/work/eeve-merged-checkpoint-500 \
    --output ./eeve-bnb-8bit
```

**예상 시간**: 각 5-10분

---

### Step 5: Hugging Face 업로드

```bash
# 양자화 모델 업로드
cd /home/work/tesseract/eeve/quant

# 4-bit 업로드
huggingface-cli upload MyeongHo0621/eeve-vss-smh-bnb-4bit \
    ./eeve-bnb-4bit \
    --repo-type model

# 8-bit 업로드
huggingface-cli upload MyeongHo0621/eeve-vss-smh-bnb-8bit \
    ./eeve-bnb-8bit \
    --repo-type model
```

---

## 📊 성능 비교

| 버전 | VRAM | 품질 | 속도 | 용도 |
|------|------|------|------|------|
| **FP16 원본** | 21GB | 100% | ⚡⚡⚡⚡ | 고사양 GPU |
| **8-bit** ⭐ | 10GB | 99.5% | ⚡⚡⚡⚡ | 프로덕션 |
| **4-bit** | 3.5GB | 98% | ⚡⚡⚡ | 개발/테스트 |

---

## 🔧 문제 해결

### 1. CUDA Out of Memory

**문제**: 훈련 중 CUDA OOM 에러

**해결**:
```python
# config.py 수정
per_device_train_batch_size = 2  # 4 → 2
gradient_accumulation_steps = 8  # 4 → 8
max_length = 1024  # 2048 → 1024
```

---

### 2. 체크포인트 로드 실패

**문제**: `conv_eeve.py`에서 어댑터 로드 실패

**해결**:
```bash
# 체크포인트 경로 확인
ls -la /home/work/eeve-korean-output/

# 올바른 경로 지정
python conv_eeve.py --checkpoint /home/work/eeve-korean-output/checkpoint-XXX
```

---

### 3. 양자화 중 에러

**문제**: `bitsandbytes` 관련 에러

**해결**:
```bash
# bitsandbytes 재설치
pip install bitsandbytes --upgrade

# CUDA 버전 확인
nvidia-smi
```

---

## 📚 관련 리소스

### 배포된 모델

| 모델 | 크기 | 링크 |
|------|------|------|
| **FP16 원본** | 21GB | [eeve-vss-smh](https://huggingface.co/MyeongHo0621/eeve-vss-smh) |
| **8-bit** ⭐ | 10GB | [eeve-vss-smh-bnb-8bit](https://huggingface.co/MyeongHo0621/eeve-vss-smh-bnb-8bit) |
| **4-bit** | 5.5GB | [eeve-vss-smh-bnb-4bit](https://huggingface.co/MyeongHo0621/eeve-vss-smh-bnb-4bit) |

### 데이터셋

| 데이터셋 | 샘플 수 | 링크 |
|----------|---------|------|
| **Korean Quality** | 54,190 | [korean-quality-cleaned](https://huggingface.co/datasets/MyeongHo0621/korean-quality-cleaned) |

### 문서

| 문서 | 설명 |
|------|------|
| `../README.md` | 프로젝트 전체 개요 |
| `../NATURAL_LLM_STRATEGY.md` | 자연스러운 LLM 생성 전략 |
| `../test_perform.py` | 모델 성능 평가 스크립트 |

---

## Best Practices

### 1. 훈련 전 체크리스트

- [ ] `config.py` 설정 확인
- [ ] 데이터셋 경로 확인
- [ ] GPU 메모리 확인 (`nvidia-smi`)
- [ ] 출력 디렉토리 확인
- [ ] 디스크 공간 확인 (최소 50GB)

### 2. 효율적인 훈련

```python
# config.py 권장 설정 (H100 기준)
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
max_length = 2048
num_train_epochs = 2
learning_rate = 1e-4
```

### 3. 체크포인트 관리

```bash
# 훈련 중 주기적으로 저장 (기본: 250 steps)
save_steps = 250

# 디스크 공간 절약을 위해 오래된 체크포인트 삭제
rm -rf /home/work/eeve-korean-output/checkpoint-{old}
```

### 4. 양자화 전략

```
개발/테스트 → 4-bit (빠른 반복)
         ↓
프로덕션 배포 → 8-bit (안정성 & 품질)
         ↓
고성능 필요 → FP16 원본
```

---

## 주요 특징

### 1. Label Masking

- 사용자 입력 부분은 loss 계산에서 제외
- Assistant 응답만 학습
- 더 자연스러운 대화 생성

### 2. EEVE 전용 템플릿

```python
template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {user_input}
Assistant: {assistant_output}"""
```

---

## 📝 참고사항

### 시스템 요구사항

| 구성 요소 | 최소 사양 | 권장 사양 |
|----------|----------|----------|
| **GPU** | A100 (40GB) | H100 (80GB) |
| **RAM** | 32GB | 64GB+ |
| **Disk** | 100GB | 500GB+ |
| **CUDA** | 11.8+ | 12.0+ |

### 라이선스

- **Base Model**: [EEVE-Korean-Instruct-10.8B](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)
- **Training Data**: CC-BY-NC-SA-4.0
- **Code**: MIT License

### Citation

```bibtex
@misc{eeve-vss-smh-2025,
  author = {MyeongHo0621},
  title = {EEVE-VSS-SMH: Fine-tuned EEVE for Korean Instructions},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/MyeongHo0621/eeve-vss-smh}}
}
```

---

## Attribute

이슈 및 개선 제안은 GitHub 또는 Hugging Face를 통해 제출해주세요.

---

**Last Updated**: 2025-10-11  
**Version**: 1.0  
**Status**: Production-Ready 


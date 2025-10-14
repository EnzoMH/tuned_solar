# Qwen 2.5-14B Korean Fine-tuning

> **Two-Stage Training Pipeline**: General Korean → WMS Domain Specialization

[![GitHub](https://img.shields.io/badge/GitHub-EnzoMH%2Fft__llm-blue)](https://github.com/EnzoMH/ft_llm)
[![Model](https://img.shields.io/badge/Model-Qwen%202.5--14B-green)](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
[![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-red)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Table of Contents

- [English Documentation](#english-documentation)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Project Structure](#project-structure)
  - [Quick Start](#quick-start)
  - [Training Details](#training-details)
- [한국어 문서](#한국어-문서)
  - [개요](#개요)
  - [주요 특징](#주요-특징)
  - [프로젝트 구조](#프로젝트-구조)
  - [빠른 시작](#빠른-시작)
  - [훈련 상세](#훈련-상세)

---

# English Documentation

## Overview

This project fine-tunes **Qwen 2.5-14B-Instruct** with **Korean-optimized tokenizer** using a two-stage training strategy:

1. **Stage 1**: General Korean language learning (54K samples)
2. **Stage 2**: WMS domain specialization (20K samples)

---

## Key Features

### **Model**
- **Qwen 2.5-14B-Instruct**: Latest model with 14B parameters
- **INT8 Quantization**: Memory-efficient training
- **LoRA Fine-tuning**: r=128, alpha=256
- **Unsloth Optimization**: 2-5x faster training

### **Dataset**
- **Stage 1**: 54,190 high-quality Korean samples (HuggingFace)
- **Stage 2**: 20,000 WMS domain QA pairs (Local)
  - EEVE template format
  - Formal Korean style (~습니다, ~입니다)
  - RAG-generated answers with domain expertise

### **Training Infrastructure**
- **GPU**: 2× NVIDIA H100 80GB HBM3
- **CPU**: 16 cores @ 192GB RAM
- **Framework**: Unsloth + PyTorch 2.8 + Transformers 4.57
- **Monitoring**: Real-time CPU/RAM/GPU tracking

---

## Project Structure

```
qwen/
├── 0_qwen_ft_us.py              # Stage 1: General Korean training
├── 1_qwen_ft_wms.py             # Stage 2: WMS domain training
├── 2_ul_hf_tknzr.py             # Korean tokenizer uploader
├── README.md                    # This file
│
├── 3_use/                       # Usage & Monitoring
│   └── monitor_training.py      # Log-based training monitor
│
├── z_util/                      # Utilities
│   ├── cpu_mntrg.py             # CPU/RAM monitoring
│   ├── gpu_mnrtg.py             # GPU/VRAM monitoring
│   └── local_dataset_loader.py  # Local JSON dataset loader
│
├── qwen-KR-14B-output-unsloth/  # Training output (Stage 1)
│   └── final/                   # Final LoRA adapters
│
└── training.log                 # Real-time training logs (not in Git)
```

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install unsloth[cu128-torch280]
pip install transformers==4.57.0
pip install datasets
pip install bitsandbytes
pip install peft
pip install trl
pip install python-dotenv

# Create .env file (DO NOT commit to Git!)
cat > .env << EOF
HF_TOKEN=your_hf_token_here
TOKENIZER=your_tokenizer_name_here
EOF
```

### Stage 1: General Korean Training

```bash
cd /home/work/tesseract/qwen
python 0_qwen_ft_us.py
```

**Training Configuration:**
- Dataset: 54,190 Korean samples (HuggingFace)
- Model: Qwen 2.5-14B-Instruct + INT8 quantization
- LoRA: r=128, alpha=256, dropout=0.0
- Batch: 4 × 4 accumulation = 16 effective
- Epochs: 3, Learning Rate: 5e-5
- Time: ~13 hours (2× H100)

**Output:** `qwen-KR-14B-output-unsloth/final/`

### Stage 2: WMS Domain Training

```bash
cd /home/work/tesseract/qwen
python 1_qwen_ft_wms.py
```

**Training Configuration:**
- Dataset: 20,000 WMS QA pairs (Local)
- Base: Stage 1 output model
- LoRA: r=64, alpha=128 (smaller for fine-tuning)
- Epochs: 2, Learning Rate: 2e-5 (lower)
- Time: ~6 hours (2× H100)

**Output:** `qwen-WMS-14B-output-unsloth/final/`

---

## Training Details

### Stage 1: General Korean

**Objective:** Build strong Korean language foundation

```python
# Key configurations
base_model = "Qwen/Qwen2.5-14B-Instruct"
tokenizer_name = os.getenv("TOKENIZER")  # From .env file
dataset_name = "your_username/korean-quality-dataset"

# LoRA
lora_r = 128
lora_alpha = 256
lora_dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
modules_to_save = ["embed_tokens", "lm_head"]  # Embedding layer training

# Training
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 5e-5
warmup_ratio = 0.1
max_seq_length = 4096
```

**Expected Loss:**
- Initial: ~2.5-4.0
- Epoch 1: ~1.5-2.5
- Epoch 3: **0.8-1.5** ✅

### Stage 2: WMS Domain

**Objective:** Specialize in WMS domain knowledge

```python
# Key configurations (differences from Stage 1)
base_model = "qwen-KR-14B-output-unsloth/final"  # Stage 1 output
dataset_name = "local"  # 20K WMS QA pairs

# LoRA (smaller)
lora_r = 64
lora_alpha = 128

# Training (gentler)
num_train_epochs = 2
learning_rate = 2e-5  # Lower LR
warmup_ratio = 0.05
```

---

## Monitoring

### Real-time Monitoring

```bash
# Watch training progress
tail -f training.log

# Check GPU usage
nvidia-smi -l 1

# Generate training plots (after training)
cd 3_use
python monitor_training.py --log ../training.log --output plots
```

### Key Metrics

- **Loss**: Should decrease from ~3.0 to ~1.0-1.5
- **Eval Loss**: Should be within 0.1-0.3 of train loss
- **GPU Utilization**: ~46% (INT8 quantization)
- **VRAM Usage**: ~18GB per GPU

---

## Troubleshooting

### 1. CUDA Out of Memory

```python
# Reduce batch size
per_device_train_batch_size = 2  # 4 → 2
gradient_accumulation_steps = 8  # 4 → 8
```

### 2. Pickle Error (dataset.map)

Already fixed in `0_qwen_ft_us.py` by using direct Python loops instead of `dataset.map()`.

### 3. Tokenizer Embedding Mismatch

Automatically handled by `resize_token_embeddings()` with smart initialization.

---

## Next Steps

### 1. Model Merging

```python
from unsloth import FastLanguageModel

model.save_pretrained_merged(
    "qwen-14B-wms-merged",
    tokenizer,
    save_method="merged_16bit"
)
```

### 2. HuggingFace Upload

```python
# Upload LoRA adapters (recommended)
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="qwen-KR-14B-output-unsloth/final",
    repo_id="your_username/qwen-14B-korean-lora",
    token="your_hf_token"
)
```

### 3. vLLM Deployment

```bash
vllm serve qwen-14B-wms-merged \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

---

## License

This project is licensed under **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

- **Allowed**: Research, Education, Personal Use
- **Prohibited**: Commercial Use
- **Required**: Attribution to original authors

Base model follows: [Qwen 2.5 License](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)

---

## Acknowledgments

- **Alibaba Qwen Team**: Qwen 2.5 model
- **Unsloth Team**: Fast training optimization
- **HuggingFace**: Transformers ecosystem

---

# 한국어 문서

## 개요

이 프로젝트는 **Qwen 2.5-14B-Instruct** 모델을 **한국어 최적화 토크나이저**와 함께 파인튜닝하는 2단계 훈련 파이프라인입니다:

1. **1단계**: 일반 한국어 학습 (54K 샘플)
2. **2단계**: WMS 도메인 특화 (20K 샘플)

---

## 주요 특징

### 🚀 **모델**
- **Qwen 2.5-14B-Instruct**: 140억 파라미터의 최신 모델
- **INT8 양자화**: 메모리 효율적인 훈련
- **LoRA 파인튜닝**: r=128, alpha=256
- **Unsloth 최적화**: 2-5배 빠른 훈련 속도

### **토크나이저**
- **한국어 최적화**: 커스텀 한국어 토크나이저 통합
- **토큰 효율**: Qwen 기본 대비 10-20% 향상
- **스마트 초기화**: 새 임베딩을 기존 평균값으로 초기화

### 📊 **데이터셋**
- **1단계**: 54,190개 고품질 한국어 샘플 (HuggingFace)
- **2단계**: 20,000개 WMS 도메인 QA 페어 (로컬)
  - EEVE 템플릿 형식
  - 격식체 한국어 (~습니다, ~입니다)
  - RAG 기반 도메인 전문 답변

### 🛠️ **훈련 인프라**
- **GPU**: 2× NVIDIA H100 80GB HBM3
- **CPU**: 16코어 @ 192GB RAM
- **프레임워크**: Unsloth + PyTorch 2.8 + Transformers 4.57
- **모니터링**: 실시간 CPU/RAM/GPU 추적

---

## 프로젝트 구조

```
qwen/
├── 0_qwen_ft_us.py              # 1단계: 일반 한국어 훈련
├── 1_qwen_ft_wms.py             # 2단계: WMS 도메인 훈련
├── 2_ul_hf_tknzr.py             # 한국어 토크나이저 업로더
├── README.md                    # 이 파일
│
├── 3_use/                       # 사용 및 모니터링
│   └── monitor_training.py      # 로그 기반 훈련 모니터
│
├── z_util/                      # 유틸리티
│   ├── cpu_mntrg.py             # CPU/RAM 모니터링
│   ├── gpu_mnrtg.py             # GPU/VRAM 모니터링
│   └── local_dataset_loader.py  # 로컬 JSON 데이터셋 로더
│
├── qwen-KR-14B-output-unsloth/  # 훈련 출력 (1단계)
│   └── final/                   # 최종 LoRA 어댑터
│
└── training.log                 # 실시간 훈련 로그 (Git 제외)
```

---

## 빠른 시작

### 사전 요구사항

```bash
# 의존성 설치
pip install unsloth[cu128-torch280]
pip install transformers==4.57.0
pip install datasets
pip install bitsandbytes
pip install peft
pip install trl
pip install python-dotenv

# .env 파일 생성 (Git에 커밋 금지!)
cat > .env << EOF
HF_TOKEN=your_hf_token_here
TOKENIZER=your_tokenizer_name_here
EOF
```

### 1단계: 일반 한국어 훈련

```bash
cd /home/work/tesseract/qwen
python 0_qwen_ft_us.py
```

**훈련 설정:**
- 데이터셋: 54,190개 한국어 샘플 (HuggingFace)
- 모델: Qwen 2.5-14B-Instruct + INT8 양자화
- LoRA: r=128, alpha=256, dropout=0.0
- 배치: 4 × 4 누적 = 16 유효 배치
- Epoch: 3, 학습률: 5e-5
- 예상 시간: 약 13시간 (H100 2개)

**출력:** `qwen-KR-14B-output-unsloth/final/`

### 2단계: WMS 도메인 훈련

```bash
cd /home/work/tesseract/qwen
python 1_qwen_ft_wms.py
```

**훈련 설정:**
- 데이터셋: 20,000개 WMS QA 페어 (로컬)
- 베이스: 1단계 출력 모델
- LoRA: r=64, alpha=128 (파인튜닝용으로 작게)
- Epoch: 2, 학습률: 2e-5 (낮게)
- 예상 시간: 약 6시간 (H100 2개)

**출력:** `qwen-WMS-14B-output-unsloth/final/`

---

## 훈련 상세

### 1단계: 일반 한국어

**목표:** 강력한 한국어 기반 구축

```python
# 주요 설정
base_model = "Qwen/Qwen2.5-14B-Instruct"
tokenizer_name = os.getenv("TOKENIZER")  # .env 파일에서 로드
dataset_name = "your_username/korean-quality-dataset"

# LoRA
lora_r = 128
lora_alpha = 256
lora_dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
modules_to_save = ["embed_tokens", "lm_head"]  # 임베딩 레이어 학습

# 훈련
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 5e-5
warmup_ratio = 0.1
max_seq_length = 4096
```

**예상 Loss:**
- 초기: ~2.5-4.0
- Epoch 1: ~1.5-2.5
- Epoch 3: **0.8-1.5** ✅

### 2단계: WMS 도메인

**목표:** WMS 도메인 지식 특화

```python
# 주요 설정 (1단계와의 차이점)
base_model = "qwen-KR-14B-output-unsloth/final"  # 1단계 출력
dataset_name = "local"  # 20K WMS QA 페어

# LoRA (작게)
lora_r = 64
lora_alpha = 128

# 훈련 (부드럽게)
num_train_epochs = 2
learning_rate = 2e-5  # 낮은 학습률
warmup_ratio = 0.05
```

---

## 모니터링

### 실시간 모니터링

```bash
# 훈련 진행 상황 확인
tail -f training.log

# GPU 사용률 확인
nvidia-smi -l 1

# 훈련 그래프 생성 (훈련 후)
cd 3_use
python monitor_training.py --log ../training.log --output plots
```

### 주요 지표

- **Loss**: ~3.0에서 ~1.0-1.5로 감소
- **Eval Loss**: Train loss의 0.1-0.3 이내
- **GPU 활용도**: ~46% (INT8 양자화)
- **VRAM 사용량**: GPU당 ~18GB

---

## 문제 해결

### 1. CUDA 메모리 부족

```python
# 배치 사이즈 감소
per_device_train_batch_size = 2  # 4 → 2
gradient_accumulation_steps = 8  # 4 → 8
```

### 2. Pickle 에러 (dataset.map)

`0_qwen_ft_us.py`에서 `dataset.map()` 대신 Python loop를 사용하여 이미 해결되었습니다.

### 3. 토크나이저 임베딩 불일치

`resize_token_embeddings()`와 스마트 초기화로 자동 처리됩니다.

---

## 다음 단계

### 1. 모델 병합

```python
from unsloth import FastLanguageModel

model.save_pretrained_merged(
    "qwen-14B-wms-merged",
    tokenizer,
    save_method="merged_16bit"
)
```

### 2. HuggingFace 업로드

```python
# LoRA 어댑터 업로드 (권장)
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="qwen-KR-14B-output-unsloth/final",
    repo_id="your_username/qwen-14B-korean-lora",
    token="your_hf_token"
)
```

### 3. vLLM 배포

```bash
vllm serve qwen-14B-wms-merged \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

---

## Git 설정 (대용량 파일 제외)

```bash
# .gitignore에 이미 설정됨
*.log
*.safetensors
*.bin
*output*/
*cache*/
```

**업로드 제외 항목:**
- 모델 체크포인트 (용량 초과)
- 캐시 파일
- 훈련 로그 (너무 큼)

**Git 푸시:**
```bash
git add qwen/*.py qwen/README.md qwen/1_guide/ qwen/3_use/
git commit -m "Update Qwen fine-tuning pipeline"
git push origin main
```

---

## 라이선스

이 프로젝트는 **CC-BY-NC-4.0** (크리에이티브 커먼즈 저작자표시-비영리 4.0 국제) 라이선스를 따릅니다.

- **허용**: 연구, 교육, 개인 사용
- **금지**: 상업적 이용
- **필수**: 원저작자 표시

베이스 모델 라이선스: [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)

---

## 감사의 말

- **Alibaba Qwen Team**: Qwen 2.5 모델
- **Unsloth Team**: 빠른 훈련 최적화
- **HuggingFace**: Transformers 생태계

---

**Last Updated**: 2025-10-14  
**Version**: 2.0.0  
**Training Status**: ✅ Stage 1 In Progress (Step 3/4827)

---

**Made with ❤️ for Korean NLP**

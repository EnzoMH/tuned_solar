#!/usr/bin/env python3
"""
Korean-Optimized Tokenizer를 Private HuggingFace Repository에 업로드

[ ! ] 회사 내부용 스크립트 - 외부 공개 금지

사용법:
    # .env 파일 생성
    cat > .env << EOF
    HF_TOKEN=hf_xxxxxxxxxxxxx
    TOKENIZER=your_tokenizer_source
    EOF
    
    python 2_ul_hf_tknzr.py
"""

import os
import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo, login

load_dotenv()

def main():
    print("\n" + "="*80)
    print(" Korean-Optimized Tokenizer Private Repository 업로드")
    print("="*80)
    
    # 1. HuggingFace 로그인
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n[ X ] 오류: HF_TOKEN 환경변수가 설정되지 않았습니다.")
        print("다음 명령어로 설정하세요:")
        print('  export HF_TOKEN="hf_xxxxxxxxxxxxx"')
        print("토큰 생성: https://huggingface.co/settings/tokens")
        return
    
    login(token=hf_token)
    print("✓ HuggingFace 로그인 완료\n")
    
    # 2. 한국어 최적화 토크나이저 다운로드
    print("="*80)
    print(" Step 1: 토크나이저 다운로드")
    print("="*80)
    
    tokenizer_source = os.getenv("TOKENIZER")
    if not tokenizer_source:
        print("\n[ X ] 오류: TOKENIZER 환경변수가 설정되지 않았습니다.")
        print(".env 파일에 TOKENIZER를 설정하세요")
        return
    
    print(f"소스: {tokenizer_source}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True
    )
    
    print(f"✓ 토크나이저 로드 완료")
    print(f"✓ Vocabulary 크기: {len(tokenizer):,} 토큰")
    print(f"✓ EOS 토큰: {tokenizer.eos_token}")
    print(f"✓ BOS 토큰: {tokenizer.bos_token}")
    print(f"✓ PAD 토큰: {tokenizer.pad_token}\n")
    
    # 3. 로컬 저장
    save_dir = "./tokenizer-temp"
    os.makedirs(save_dir, exist_ok=True)
    
    tokenizer.save_pretrained(save_dir)
    print(f"✓ 로컬 저장 완료: {save_dir}\n")
    
    # 저장된 파일 확인
    files = os.listdir(save_dir)
    print("저장된 파일:")
    for f in sorted(files):
        file_path = os.path.join(save_dir, f)
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {f:30s} ({size:.1f} KB)")
    print()
    
    # 4. Private Repository 생성
    print("="*80)
    print(" Step 2: Private Repository 생성")
    print("="*80)
    
    repo_name = "MyeongHo0621/tokenizer-private"
    print(f"Repository: {repo_name}")
    
    try:
        create_repo(
            repo_id=repo_name,
            private=True,  # ⭐ Private 설정
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository 생성 완료 (Private)\n")
    except Exception as e:
        print(f"[ ! ] Repository 생성 경고: {e}")
        print("(이미 존재하면 무시됩니다)\n")
    
    # 5. 토크나이저 업로드
    print("="*80)
    print(" Step 3: 토크나이저 업로드")
    print("="*80)
    
    print("업로드 중...")
    tokenizer.push_to_hub(
        repo_name,
        private=True,
        commit_message="Upload Private Tokenizer (Private)"
    )
    
    print(f"✓ 업로드 완료!\n")
    
    # 6. README 업데이트
    print("="*80)
    print(" Step 4: README 작성")
    print("="*80)
    
    readme_content = f"""---
language:
- ko
- en
library_name: transformers
license: cc-by-nc-4.0
tags:
- tokenizer
- private
- korean
- non-commercial
---

# Korean-Optimized Tokenizer (Private)

한국어 최적화 토크나이저입니다.

## [ * ] 특징

- **Vocabulary 크기**: {len(tokenizer):,} 토큰
- **언어**: 한국어-영어 이중언어 최적화
- **한국어 효율**: 기존 토크나이저 대비 토큰 수 절감

## [ SECURITY ] 접근 권한

이 토크나이저는 **Private Repository**입니다.
- 본인만 접근 가능
- HuggingFace 토큰 필요

## [ CODE ] 사용법

### **기본 사용**

```python
from transformers import AutoTokenizer
from huggingface_hub import login

# 1. 로그인 (본인의 HF 토큰)
login(token="hf_xxxxxxxxxxxxx")

# 2. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "{repo_name}",
    token="hf_xxxxxxxxxxxxx",
    trust_remote_code=True
)

# 3. 사용 예시
text = "재고 회전율을 개선하는 방법은?"
tokens = tokenizer(text, return_tensors="pt")
print(f"토큰 개수: {{tokens['input_ids'].shape[1]}}")

# 디코딩
decoded = tokenizer.decode(tokens['input_ids'][0])
print(decoded)
```

### **Qwen 모델과 함께 사용**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

# 로그인
login(token="hf_xxxxxxxxxxxxx")

# Qwen 모델 로드 (예시: Qwen2.5-14B)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 한국어 최적화 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "{repo_name}",
    token="hf_xxxxxxxxxxxxx",
    trust_remote_code=True
)

# 임베딩 크기 조정
model.resize_token_embeddings(len(tokenizer))

# 추론
messages = [
    {{"role": "system", "content": "당신은 물류 시스템 전문가입니다."}},
    {{"role": "user", "content": "WMS란 무엇인가요?"}}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## [ * ] 토큰 통계

```python
# 한국어 텍스트 토큰화 효율성
test_texts = [
    "재고 회전율 개선 방법",
    "옴니채널 재고 통합 전략",
    "WMS 피킹 최적화"
]

for text in test_texts:
    tokens = tokenizer(text)['input_ids']
    print(f"'{{text}}': {{len(tokens)}} 토큰")
```

## [ ! ] 주의사항

- 이 토크나이저는 **Private**입니다.
- 접근하려면 **본인의 HuggingFace 토큰**이 필요합니다.
- 다른 모델과 사용 시 `resize_token_embeddings()` 필수

## [ LICENSE ] 라이선스

**CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International)

- **허용**: 연구, 교육, 개인 사용
- **금지**: 상업적 이용
- **필수**: 원저작자 표시

## [ * ] 관련 링크

- [HuggingFace 토큰 생성](https://huggingface.co/settings/tokens)

---

**Last Updated**: {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
    
    api = HfApi()
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_name,
    )
    
    print("✓ README 업로드 완료\n")
    
    # 7. 완료 메시지
    print("="*80)
    print(" [ OK ] 모든 작업 완료!")
    print("="*80)
    print(f"\n [ * ] Repository: https://huggingface.co/{repo_name}")
    print(f" [ SECURITY ] Private: 본인만 접근 가능")
    print(f" [ * ] Vocabulary: {len(tokenizer):,} 토큰")
    print(f"\n사용법:")
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{repo_name}", token="hf_xxx")')
    print("\n" + "="*80 + "\n")
    
    # 정리
    import shutil
    shutil.rmtree(save_dir)
    print(f"✓ 임시 파일 정리 완료: {save_dir}")

if __name__ == "__main__":
    main()


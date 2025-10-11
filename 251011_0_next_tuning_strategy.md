# **코드 생성 + JSON 출력 + 고급 튜닝 전략**

## **1. 코드 생성 특화 (Python/C++/UE)**

### **전략 A: Instruction Tuning with Code Dataset**

```python
# 데이터셋 구성 (예시)
code_dataset = [
    {
        "instruction": "다음 기능을 Python으로 구현하세요: 이진 탐색 알고리즘",
        "output": """```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```"""
    },
    {
        "instruction": "UE5에서 캐릭터 이동을 C++로 구현하세요",
        "output": """```cpp
void AMyCharacter::MoveForward(float Value)
{
    if (Controller && Value != 0.0f)
    {
        const FRotator Rotation = Controller->GetControlRotation();
        const FRotator YawRotation(0, Rotation.Yaw, 0);
        const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
        AddMovementInput(Direction, Value);
    }
}
```"""
    },
    {
        "instruction": "JSON 형식으로 사용자 정보를 출력하세요",
        "output": """```json
{
  "user": {
    "id": 12345,
    "name": "홍길동",
    "email": "hong@example.com",
    "roles": ["admin", "developer"],
    "active": true
  }
}
```"""
    }
]
```

### **데이터 소스**

```python
# Hugging Face에서 코드 데이터 다운로드
from datasets import load_dataset, concatenate_datasets

# 1. Python 코드
code_python = load_dataset("iamtarun/python_code_instructions_18k_alpaca")

# 2. 다국어 코드 (C++ 포함)
code_multi = load_dataset("bigcode/starcoderdata", split="train[:50000]")

# 3. JSON 생성 데이터 (직접 제작 or GPT로 생성)
# OpenAI API로 JSON 샘플 대량 생성 가능

# 4. UE5 C++ (GitHub 크롤링 or 직접 제작)
# 예: https://github.com/EpicGames/UnrealEngine 문서 기반

# 합치기
combined_code = concatenate_datasets([
    korean_instruction_54k,  # 기존 데이터
    code_python,             # 18K
    code_multi.select(range(20000)),  # 20K
    json_custom_5k,          # 5K (직접 제작)
    ue_cpp_3k               # 3K (직접 제작)
])
# 총: 100K
```

---

## **2. JSON 출력 특화 훈련**

### **Structured Output Format**

```python
# 프롬프트 템플릿
SYSTEM_PROMPT = """You are a helpful AI assistant that provides structured outputs.
When asked for JSON, always output valid JSON format with proper syntax."""

# 훈련 데이터 예시
json_training_data = [
    {
        "instruction": "사용자 정보를 JSON으로 만들어주세요. 이름: 김철수, 나이: 28, 직업: 개발자",
        "output": """{
  "name": "김철수",
  "age": 28,
  "occupation": "개발자",
  "skills": ["Python", "JavaScript", "C++"],
  "experience_years": 5
}"""
    },
    {
        "instruction": "API 에러 응답을 JSON 형식으로 작성하세요",
        "output": """{
  "error": {
    "code": 404,
    "message": "Resource not found",
    "timestamp": "2025-10-11T12:00:00Z",
    "path": "/api/users/12345"
  }
}"""
    },
    # ... 수천 개 추가
]
```

### **Constrained Generation (출력 제약)**

```python
# inference 시 JSON 검증
import json

def generate_json(model, tokenizer, prompt):
    outputs = model.generate(
        **tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=2048,
        temperature=0.3,  # 낮게 (일관성)
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # JSON 추출 및 검증
    try:
        # ```json ... ``` 블록 추출
        json_str = response.split("```json")[1].split("```")[0].strip()
        parsed = json.loads(json_str)
        return parsed
    except:
        return {"error": "Invalid JSON generated"}
```

---

## **3. 고급 파인튜닝 전략**

### **Option A: Context Engineering (Prompt Engineering)**

```python
# Few-shot 예제 포함
CONTEXT_TEMPLATE = """You are an expert programmer specializing in Python, C++, and Unreal Engine.

Examples:
1. User: "Python으로 피보나치 수열 구현"
   Assistant: 
   ```python
   def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n-1) + fibonacci(n-2)
   ```

2. User: "JSON 형식으로 게임 캐릭터 정보"
   Assistant:
   ```json
   {{
     "character": {{
       "name": "Hero",
       "level": 10,
       "stats": {{"hp": 100, "mp": 50}}
     }}
   }}
   ```

Now answer the user's question:
{user_question}
"""

# 훈련 데이터에 이 컨텍스트 포함
def add_context(example):
    example['input'] = CONTEXT_TEMPLATE.format(
        user_question=example['instruction']
    )
    return example

dataset = dataset.map(add_context)
```

---

### **Option B: RLHF (Reinforcement Learning from Human Feedback)**

```python
# 1단계: SFT (Supervised Fine-Tuning) - 이미 진행 중
# 2단계: Reward Model 학습
# 3단계: PPO/DPO로 정책 최적화

from trl import DPOTrainer, DPOConfig

# DPO 데이터 형식
dpo_dataset = [
    {
        "prompt": "Python 리스트 컴프리헨션 예제",
        "chosen": "[x**2 for x in range(10)]",  # 좋은 답변
        "rejected": "for 루프 써서..."  # 나쁜 답변
    }
]

# DPO Trainer 설정
dpo_config = DPOConfig(
    beta=0.1,  # KL divergence 가중치
    learning_rate=5e-7,
    per_device_train_batch_size=4,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 원본 모델 (reference)
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

**DPO 장점:**
- ✅ PPO보다 간단 (reward model 불필요)
- ✅ 안정적 학습
- ✅ 선호하는 스타일로 유도 (JSON, 코드 품질 등)

---

### **Option C: Reasoning Model (CoT - Chain of Thought)**

```python
# Reasoning 데이터 형식
reasoning_dataset = [
    {
        "instruction": "피보나치 수열을 재귀가 아닌 동적 프로그래밍으로 구현하는 이유를 설명하고 코드를 작성하세요",
        "output": """<think>
재귀 방식의 문제점:
1. 시간 복잡도: O(2^n) - 지수적 증가
2. 중복 계산이 많음 (예: fib(5) 계산 시 fib(3)을 여러 번 계산)
3. 스택 오버플로우 위험

동적 프로그래밍의 장점:
1. 시간 복잡도: O(n) - 선형
2. 메모이제이션으로 중복 계산 제거
3. 반복문 사용으로 스택 안전
</think>

따라서 다음과 같이 구현합니다:

```python
def fibonacci_dp(n):
    if n <= 1:
        return n
    
    # DP 테이블
    dp = [0] * (n + 1)
    dp[1] = 1
    
    # 상향식 계산
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# 공간 최적화 버전
def fibonacci_optimized(n):
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1
```

시간 복잡도: O(n)
공간 복잡도: O(1) (최적화 버전)
"""
    }
]

# System prompt에 reasoning 활성화
REASONING_PROMPT = """You are an expert AI assistant that thinks step-by-step.
Before providing the final answer, explain your reasoning process inside <think> tags.

Example:
<think>
1. Analyze the problem
2. Consider alternatives
3. Choose best approach
</think>

Your answer here...
"""
```

---

## **4. 통합 훈련 전략 (Multi-Stage)**

```python
# Stage 1: Base Instruction Tuning (2 epochs)
#   - 기존 54K 한국어 instruction
#   - Learning rate: 1e-4

# Stage 2: Code Specialization (1 epoch)
#   - Python/C++/JSON 데이터 추가 (50K)
#   - Learning rate: 5e-5 (낮게)

# Stage 3: DPO (Optional, 0.5 epoch)
#   - 코드 품질, JSON 형식 선호
#   - Learning rate: 5e-7

# Stage 4: Reasoning (Optional, 0.5 epoch)
#   - CoT 데이터 추가 (10K)
#   - Learning rate: 3e-5
```

### **코드 예시**

```python
#!/usr/bin/env python3
"""
EEVE 다목적 파인튜닝: 한국어 + 코드 + JSON + Reasoning
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

# 1. 데이터 준비
def prepare_multitask_dataset():
    # 기본 한국어
    korean = load_dataset("your_korean_54k")
    
    # 코드 데이터
    code_python = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
    
    # JSON 데이터 (직접 생성 or GPT로 생성)
    json_data = load_custom_json_dataset()
    
    # C++/UE 데이터 (직접 수집)
    cpp_ue = load_custom_cpp_dataset()
    
    # Reasoning 데이터
    reasoning = load_custom_reasoning_dataset()
    
    # 통합
    combined = concatenate_datasets([
        korean['train'],           # 54K
        code_python['train'],      # 18K
        json_data,                 # 5K
        cpp_ue,                    # 3K
        reasoning                  # 10K
    ])
    # 총: 90K
    
    return combined

# 2. Stage별 훈련
def multi_stage_training():
    model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    
    # Stage 1: Base Instruction
    train_stage1(
        model_id=model_id,
        dataset=korean_only,
        epochs=2,
        lr=1e-4,
        output="./stage1_base"
    )
    
    # Stage 2: Code Specialization
    train_stage2(
        model_id="./stage1_base",  # Stage 1 결과 로드
        dataset=code_json_dataset,
        epochs=1,
        lr=5e-5,
        output="./stage2_code"
    )
    
    # Stage 3: DPO (선택)
    if use_dpo:
        train_dpo(
            model_id="./stage2_code",
            dpo_dataset=preference_data,
            output="./stage3_dpo"
        )
    
    # Stage 4: Reasoning (선택)
    if use_reasoning:
        train_reasoning(
            model_id="./stage3_dpo",
            reasoning_dataset=cot_data,
            output="./final_model"
        )

# 3. 특수 프롬프트 템플릿
CODE_SYSTEM_PROMPT = """You are an expert programmer in Python, C++, and Unreal Engine.
When asked for code, provide clean, well-commented, production-ready code.
When asked for JSON, always output valid JSON with proper formatting."""

REASONING_SYSTEM_PROMPT = """Think step-by-step before answering.
Wrap your reasoning in <think> tags, then provide the final answer."""

# 4. Inference (사용 예시)
def generate_code(prompt, language="python"):
    full_prompt = f"{CODE_SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
    
    outputs = model.generate(
        **tokenizer(full_prompt, return_tensors="pt").to(model.device),
        max_new_tokens=4096,  # 긴 코드 가능
        temperature=0.2,      # 낮게 (일관성)
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 예시
result = generate_code("Python으로 이진 트리 구현", "python")
print(result)
```

---

## **5. 추천 전략 (우선순위)**

### **Phase 1: 기본 (필수)**
```python
✅ Stage 1: Instruction Tuning (54K Korean)
✅ Stage 2: Code Specialization (50K Code/JSON)
   - 총 104K 데이터
   - 3 epochs
   - 예상 시간: 6-10시간 (H100)
```

### **Phase 2: 고급 (선택)**
```python
⭐ DPO: 코드 품질 향상
   - 좋은 코드 vs 나쁜 코드 preference
   - 1000-5000 샘플로도 효과적

⭐ Reasoning: 복잡한 문제 해결
   - CoT 데이터 10K
   - 1 epoch
```

### **Phase 3: 평가**
```python
# HumanEval (Python)
# MBPP (Python)
# Custom UE C++ benchmark
# JSON validation rate
# 한국어 instruction following
```

---

## **6. 데이터 생성 팁**

### **GPT-4o로 대량 생성**
```python
# OpenAI API로 JSON/코드 데이터 생성
import openai

def generate_training_data(prompt_template, n=1000):
    dataset = []
    for i in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "Generate training examples for code/JSON"
            }, {
                "role": "user",
                "content": prompt_template
            }]
        )
        dataset.append(parse_response(response))
    return dataset

# 예시
json_data = generate_training_data(
    "Create 10 examples of user asking for JSON and correct JSON response",
    n=500
)
```

---

## **7. 최종 추천 구성**

```python
# 데이터 믹스
korean_instruction: 54K   (54%)
python_code: 18K          (18%)
cpp_ue_code: 5K           (5%)
json_output: 10K          (10%)
reasoning_cot: 10K        (10%)
dpo_preference: 3K        (3%)
------------------------
Total: 100K

# 훈련 설정
- Epochs: 3 (base) + 1 (DPO)
- Learning rate: 1e-4 → 5e-5 → 5e-7
- Batch size: 8 (H100)
- 예상 시간: 8-12시간
```

---

1. ✅ **Continual Pretraining 불필요** (EEVE가 이미 우수)
2. ✅ **Instruction + Code 특화** (Python/C++/JSON)
3. ✅ **DPO로 품질 향상** (선택)
4. ✅ **CoT로 Reasoning 강화** (선택)
5. ✅ **8K 출력 활용** (긴 코드 생성 가능)

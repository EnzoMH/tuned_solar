"""
H100 GPU 최적화 유틸리티
SOLID 원칙 기반 설계
"""

import time
import torch
from typing import Optional, Dict, List
from contextlib import contextmanager


# ============================================================
# 1. GPU 타이머 (Single Responsibility)
# ============================================================
class GPUTimer:
    """정확한 GPU 시간 측정 (CUDA synchronization 포함)"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.use_cuda = torch.cuda.is_available() and "cuda" in device
    
    @contextmanager
    def measure(self):
        """
        컨텍스트 매니저로 시간 측정
        
        Usage:
            timer = GPUTimer()
            with timer.measure() as t:
                # GPU 작업
                pass
            print(f"Time: {t.elapsed:.3f}s")
        """
        if self.use_cuda:
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        result = {"elapsed": 0.0}
        
        try:
            yield result
        finally:
            if self.use_cuda:
                torch.cuda.synchronize()
            result["elapsed"] = time.perf_counter() - start_time
    
    def measure_sync(self, func, *args, **kwargs):
        """
        함수 실행 시간 측정 (동기화 포함)
        
        Returns:
            (result, elapsed_time)
        """
        if self.use_cuda:
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        return result, elapsed


# ============================================================
# 2. GPU 모니터 (Single Responsibility)
# ============================================================
class GPUMonitor:
    """GPU 사용량 모니터링"""
    
    @staticmethod
    def get_memory_info(device_id: int = 0) -> Dict[str, float]:
        """
        GPU 메모리 정보 조회
        
        Returns:
            {
                'total_gb': 80.0,
                'used_gb': 2.5,
                'free_gb': 77.5,
                'utilization_pct': 3.1
            }
        """
        if not torch.cuda.is_available():
            return {}
        
        try:
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            free, total_mem = torch.cuda.mem_get_info(device_id)
            used = (total_mem - free) / (1024**3)
            free_gb = free / (1024**3)
            
            return {
                'total_gb': round(total, 2),
                'used_gb': round(used, 2),
                'free_gb': round(free_gb, 2),
                'utilization_pct': round((used / total) * 100, 1)
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def print_all_gpus():
        """모든 GPU 상태 출력"""
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return
        
        print(f"\n{'='*70}")
        print("GPU Status:")
        print(f"{'='*70}")
        
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            info = GPUMonitor.get_memory_info(i)
            
            print(f"GPU {i}: {name}")
            if 'error' not in info:
                print(f"  Memory: {info['used_gb']:.1f}/{info['total_gb']:.1f} GB ({info['utilization_pct']}%)")
                print(f"  Free: {info['free_gb']:.1f} GB")
            else:
                print(f"  Error: {info['error']}")
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def get_optimal_batch_size(
        model_memory_gb: float,
        available_memory_gb: float,
        safety_factor: float = 0.8
    ) -> int:
        """
        최적 배치 사이즈 계산
        
        Args:
            model_memory_gb: 모델 1개당 메모리 (GB)
            available_memory_gb: 사용 가능한 GPU 메모리 (GB)
            safety_factor: 안전 마진 (기본 80%)
        
        Returns:
            권장 배치 크기
        """
        usable_memory = available_memory_gb * safety_factor
        batch_size = int(usable_memory / model_memory_gb)
        return max(1, batch_size)


# ============================================================
# 3. Multi-GPU 배치 생성기 (Strategy Pattern)
# ============================================================
class MultiGPUBatchGenerator:
    """H100 2개를 활용한 병렬 배치 생성"""
    
    def __init__(
        self,
        model,
        tokenizer,
        num_gpus: int = 2,
        max_batch_size: int = 16
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.max_batch_size = max_batch_size
        self.timer = GPUTimer()
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        **gen_kwargs
    ) -> List[str]:
        """
        배치로 한번에 생성 (단일 GPU)
        
        Args:
            prompts: 프롬프트 리스트
            max_new_tokens: 최대 생성 토큰 수
            
        Returns:
            생성된 텍스트 리스트
        """
        # 배치 토크나이징
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048
        ).to(self.model.device)
        
        # 배치 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **gen_kwargs
            )
        
        # 디코딩
        generated_texts = []
        for i, output in enumerate(outputs):
            input_len = inputs.input_ids[i].shape[0]
            generated = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            )
            generated_texts.append(generated)
        
        return generated_texts
    
    def split_and_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512
    ) -> tuple[List[str], float]:
        """
        프롬프트를 배치로 나눠서 생성
        
        Returns:
            (generated_texts, total_time)
        """
        all_results = []
        total_time = 0.0
        
        for i in range(0, len(prompts), self.max_batch_size):
            batch = prompts[i:i + self.max_batch_size]
            
            batch_results, elapsed = self.timer.measure_sync(
                self.generate_batch,
                batch,
                max_new_tokens
            )
            
            all_results.extend(batch_results)
            total_time += elapsed
            
            print(f"✓ Batch {i//self.max_batch_size + 1}: {len(batch)} prompts in {elapsed:.2f}s", flush=True)
        
        return all_results, total_time


# ============================================================
# 4. vLLM 래퍼 (Adapter Pattern) - 선택적 사용
# ============================================================
class vLLMAnswerMaker:
    """
    vLLM 기반 H100 최적화 답변 생성기
    
    Requirements:
        pip install vllm
    """
    
    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.85
    ):
        """
        Args:
            model_name: 모델 이름
            tensor_parallel_size: 사용할 GPU 수 (1 or 2)
            gpu_memory_utilization: GPU 메모리 사용률 (0.0-1.0)
        """
        try:
            from vllm import LLM, SamplingParams
            self.vllm_available = True
        except ImportError:
            self.vllm_available = False
            print("⚠️ vLLM not installed. Use: pip install vllm")
            return
        
        print(f"Loading {model_name} with vLLM (TP={tensor_parallel_size})...", flush=True)
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1500,
            top_p=0.95
        )
        
        self.timer = GPUTimer()
        
        print(f"✓ vLLM loaded on {tensor_parallel_size} GPU(s)\n", flush=True)
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 1500
    ) -> tuple[List[str], float]:
        """
        초고속 배치 생성
        
        Returns:
            (generated_texts, elapsed_time)
        """
        if not self.vllm_available:
            raise RuntimeError("vLLM not available")
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            top_p=0.95
        )
        
        outputs, elapsed = self.timer.measure_sync(
            self.llm.generate,
            prompts,
            sampling_params
        )
        
        texts = [output.outputs[0].text for output in outputs]
        
        return texts, elapsed
    
    @staticmethod
    def is_available() -> bool:
        """vLLM 사용 가능 여부"""
        try:
            import vllm
            return True
        except ImportError:
            return False


# ============================================================
# 5. 팩토리 함수 (Factory Pattern)
# ============================================================
def create_optimized_generator(
    model,
    tokenizer,
    use_vllm: bool = False,
    **kwargs
):
    """
    최적화된 생성기 생성
    
    Args:
        model: Hugging Face 모델
        tokenizer: 토크나이저
        use_vllm: vLLM 사용 여부
        
    Returns:
        MultiGPUBatchGenerator 또는 vLLMAnswerMaker
    """
    if use_vllm and vLLMAnswerMaker.is_available():
        print("Using vLLM (H100 optimized)")
        return vLLMAnswerMaker(**kwargs)
    else:
        print("Using standard batch generator")
        return MultiGPUBatchGenerator(model, tokenizer, **kwargs)


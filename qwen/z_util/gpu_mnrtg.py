#!/usr/bin/env python3
"""
GPU 모니터링 유틸리티
- 실시간 GPU 메모리 추적
- GPU 사용률 추적
- Multi-GPU 지원
- CUDA 메모리 상세 추적
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import subprocess
import re


@dataclass
class GPUSnapshot:
    """GPU 상태 스냅샷"""
    timestamp: str
    gpu_id: int
    name: str
    memory_allocated_gb: float
    memory_reserved_gb: float
    memory_total_gb: float
    memory_percent: float
    utilization_percent: float
    temperature: Optional[float]
    power_usage: Optional[float]
    power_limit: Optional[float]


class GPUMonitor:
    """GPU 모니터링 클래스"""
    
    def __init__(self):
        """GPU 모니터 초기화"""
        self.available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.available else 0
        self.snapshots: List[GPUSnapshot] = []
        
        if self.available:
            self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
            self.device_properties = [torch.cuda.get_device_properties(i) for i in range(self.device_count)]
        else:
            self.device_names = []
            self.device_properties = []
    
    def get_snapshot(self, gpu_id: int = 0) -> Optional[GPUSnapshot]:
        """특정 GPU의 현재 상태 스냅샷"""
        if not self.available or gpu_id >= self.device_count:
            return None
        
        # PyTorch 메모리 정보
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        total = self.device_properties[gpu_id].total_memory / (1024**3)
        memory_percent = (allocated / total) * 100 if total > 0 else 0.0
        
        # nvidia-smi 정보 (utilization, temperature, power)
        utilization, temperature, power_usage, power_limit = self._get_nvidia_smi_info(gpu_id)
        
        snapshot = GPUSnapshot(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            gpu_id=gpu_id,
            name=self.device_names[gpu_id],
            memory_allocated_gb=allocated,
            memory_reserved_gb=reserved,
            memory_total_gb=total,
            memory_percent=memory_percent,
            utilization_percent=utilization,
            temperature=temperature,
            power_usage=power_usage,
            power_limit=power_limit
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_all_snapshots(self) -> List[GPUSnapshot]:
        """모든 GPU의 스냅샷"""
        snapshots = []
        for i in range(self.device_count):
            snapshot = self.get_snapshot(i)
            if snapshot:
                snapshots.append(snapshot)
        return snapshots
    
    def _get_nvidia_smi_info(self, gpu_id: int) -> tuple:
        """nvidia-smi에서 추가 정보 가져오기"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit', 
                 '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                utilization = float(values[0].strip())
                temperature = float(values[1].strip())
                power_usage = float(values[2].strip())
                power_limit = float(values[3].strip())
                return utilization, temperature, power_usage, power_limit
        except Exception:
            pass
        
        return 0.0, None, None, None
    
    def get_memory_summary(self, gpu_id: int = 0) -> Dict:
        """CUDA 메모리 상세 정보"""
        if not self.available or gpu_id >= self.device_count:
            return {}
        
        # 메모리 통계
        allocated = torch.cuda.memory_allocated(gpu_id)
        reserved = torch.cuda.memory_reserved(gpu_id)
        max_allocated = torch.cuda.max_memory_allocated(gpu_id)
        max_reserved = torch.cuda.max_memory_reserved(gpu_id)
        
        # 메모리 통계 (상세)
        try:
            mem_stats = torch.cuda.memory_stats(gpu_id)
            active_bytes = mem_stats.get('active_bytes.all.current', 0)
            inactive_bytes = mem_stats.get('inactive_split_bytes.all.current', 0)
            allocated_blocks = mem_stats.get('allocated_blocks.all.current', 0)
        except:
            active_bytes = 0
            inactive_bytes = 0
            allocated_blocks = 0
        
        return {
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'max_allocated_gb': max_allocated / (1024**3),
            'max_reserved_gb': max_reserved / (1024**3),
            'active_gb': active_bytes / (1024**3),
            'inactive_gb': inactive_bytes / (1024**3),
            'allocated_blocks': allocated_blocks,
            'fragmentation': (reserved - allocated) / reserved * 100 if reserved > 0 else 0.0
        }
    
    def log_snapshot(self, logger, gpu_id: int = 0, stage: str = ""):
        """스냅샷을 로그로 출력"""
        snapshot = self.get_snapshot(gpu_id)
        if not snapshot:
            logger.warning(f"GPU {gpu_id} 정보를 가져올 수 없습니다")
            return None
        
        logger.info("="*80)
        logger.info(f"[{stage}] GPU{gpu_id} 상세 모니터링")
        logger.info(f"  시간: {snapshot.timestamp}")
        logger.info(f"  GPU: {snapshot.name}")
        logger.info(f"  메모리 할당: {snapshot.memory_allocated_gb:.2f}GB / {snapshot.memory_total_gb:.2f}GB ({snapshot.memory_percent:.1f}%)")
        logger.info(f"  메모리 예약: {snapshot.memory_reserved_gb:.2f}GB")
        logger.info(f"  GPU 사용률: {snapshot.utilization_percent:.1f}%")
        
        if snapshot.temperature is not None:
            logger.info(f"  온도: {snapshot.temperature:.0f}°C")
        
        if snapshot.power_usage is not None and snapshot.power_limit is not None:
            logger.info(f"  전력: {snapshot.power_usage:.0f}W / {snapshot.power_limit:.0f}W")
        
        # 메모리 상세 정보
        mem_summary = self.get_memory_summary(gpu_id)
        if mem_summary:
            logger.info(f"  메모리 단편화: {mem_summary['fragmentation']:.1f}%")
            logger.info(f"  활성 메모리: {mem_summary['active_gb']:.2f}GB")
            logger.info(f"  비활성 메모리: {mem_summary['inactive_gb']:.2f}GB")
        
        logger.info("="*80)
        
        return snapshot
    
    def log_all_gpus(self, logger, stage: str = ""):
        """모든 GPU 로그"""
        if not self.available:
            logger.warning("CUDA를 사용할 수 없습니다")
            return
        
        logger.info("="*80)
        logger.info(f"[{stage}] 전체 GPU 상태")
        logger.info(f"  총 GPU 수: {self.device_count}")
        
        for i in range(self.device_count):
            snapshot = self.get_snapshot(i)
            if snapshot:
                logger.info(f"  GPU{i} ({snapshot.name}): "
                           f"{snapshot.memory_allocated_gb:.1f}GB/{snapshot.memory_total_gb:.1f}GB "
                           f"({snapshot.memory_percent:.1f}%) | "
                           f"Load: {snapshot.utilization_percent:.0f}%")
        
        logger.info("="*80)
    
    def check_memory_available(self, gpu_id: int = 0, required_gb: float = 1.0) -> bool:
        """특정 GPU에 충분한 메모리가 있는지 확인"""
        if not self.available or gpu_id >= self.device_count:
            return False
        
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        total = self.device_properties[gpu_id].total_memory / (1024**3)
        available = total - allocated
        
        return available >= required_gb
    
    def clear_cache(self, gpu_id: Optional[int] = None):
        """GPU 캐시 정리"""
        if not self.available:
            return
        
        if gpu_id is not None:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    
    def get_summary(self, gpu_id: int = 0) -> Dict:
        """모니터링 요약 통계"""
        gpu_snapshots = [s for s in self.snapshots if s.gpu_id == gpu_id]
        
        if not gpu_snapshots:
            return {}
        
        memory_percents = [s.memory_percent for s in gpu_snapshots]
        utilizations = [s.utilization_percent for s in gpu_snapshots]
        
        return {
            'memory_avg': sum(memory_percents) / len(memory_percents),
            'memory_max': max(memory_percents),
            'memory_min': min(memory_percents),
            'utilization_avg': sum(utilizations) / len(utilizations),
            'utilization_max': max(utilizations),
            'utilization_min': min(utilizations),
            'snapshots_count': len(gpu_snapshots)
        }


if __name__ == "__main__":
    # 테스트
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    monitor = GPUMonitor()
    
    if monitor.available:
        monitor.log_all_gpus(logger, "테스트")
        
        print("\nGPU 0 메모리 상세:")
        import json
        print(json.dumps(monitor.get_memory_summary(0), indent=2))
    else:
        print("CUDA를 사용할 수 없습니다")


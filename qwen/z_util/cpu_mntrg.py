#!/usr/bin/env python3
"""
CPU 모니터링 유틸리티
- 실시간 CPU 사용률 추적
- 메모리 사용량 추적
- 프로세스별 리소스 추적
"""

import psutil
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CPUSnapshot:
    """CPU 상태 스냅샷"""
    timestamp: str
    cpu_percent: float
    cpu_count: int
    cpu_freq_current: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    swap_used_gb: float
    swap_total_gb: float
    process_cpu_percent: float
    process_ram_gb: float
    process_threads: int


class CPUMonitor:
    """CPU 및 메모리 모니터링 클래스"""
    
    def __init__(self, process_name: Optional[str] = None):
        """
        Args:
            process_name: 모니터링할 프로세스 이름 (None이면 현재 프로세스)
        """
        self.process_name = process_name
        self.process = psutil.Process(os.getpid()) if process_name is None else None
        self.snapshots: List[CPUSnapshot] = []
        
    def get_snapshot(self) -> CPUSnapshot:
        """현재 CPU/메모리 상태 스냅샷"""
        # CPU 정보
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_current = cpu_freq.current if cpu_freq else 0.0
        
        # RAM 정보
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent
        
        # Swap 정보
        swap = psutil.swap_memory()
        swap_used_gb = swap.used / (1024**3)
        swap_total_gb = swap.total / (1024**3)
        
        # 프로세스 정보
        if self.process:
            try:
                process_cpu_percent = self.process.cpu_percent(interval=0.1)
                process_ram_gb = self.process.memory_info().rss / (1024**3)
                process_threads = self.process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_cpu_percent = 0.0
                process_ram_gb = 0.0
                process_threads = 0
        else:
            process_cpu_percent = 0.0
            process_ram_gb = 0.0
            process_threads = 0
        
        snapshot = CPUSnapshot(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_freq_current=cpu_freq_current,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            ram_percent=ram_percent,
            swap_used_gb=swap_used_gb,
            swap_total_gb=swap_total_gb,
            process_cpu_percent=process_cpu_percent,
            process_ram_gb=process_ram_gb,
            process_threads=process_threads
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_detailed_info(self) -> Dict:
        """상세 CPU 정보"""
        cpu_times = psutil.cpu_times()
        cpu_stats = psutil.cpu_stats()
        
        info = {
            'cpu_times': {
                'user': cpu_times.user,
                'system': cpu_times.system,
                'idle': cpu_times.idle,
                'iowait': getattr(cpu_times, 'iowait', 0.0),
            },
            'cpu_stats': {
                'ctx_switches': cpu_stats.ctx_switches,
                'interrupts': cpu_stats.interrupts,
                'soft_interrupts': cpu_stats.soft_interrupts,
            },
            'cpu_count': {
                'logical': psutil.cpu_count(logical=True),
                'physical': psutil.cpu_count(logical=False),
            }
        }
        
        return info
    
    def get_top_processes(self, n: int = 5) -> List[Dict]:
        """CPU/메모리 사용량 상위 프로세스"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                processes.append({
                    'pid': info['pid'],
                    'name': info['name'],
                    'cpu_percent': info['cpu_percent'],
                    'memory_percent': info['memory_percent']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # CPU 사용률 기준 정렬
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        return processes[:n]
    
    def log_snapshot(self, logger, stage: str = ""):
        """스냅샷을 로그로 출력"""
        snapshot = self.get_snapshot()
        
        logger.info("="*80)
        logger.info(f"[{stage}] CPU/RAM 상세 모니터링")
        logger.info(f"  시간: {snapshot.timestamp}")
        logger.info(f"  CPU 사용률: {snapshot.cpu_percent:.1f}% ({snapshot.cpu_count} cores @ {snapshot.cpu_freq_current:.0f} MHz)")
        logger.info(f"  RAM: {snapshot.ram_used_gb:.1f}GB / {snapshot.ram_total_gb:.1f}GB ({snapshot.ram_percent:.1f}%)")
        
        if snapshot.swap_total_gb > 0:
            logger.info(f"  Swap: {snapshot.swap_used_gb:.1f}GB / {snapshot.swap_total_gb:.1f}GB")
        
        if self.process:
            logger.info(f"  프로세스 CPU: {snapshot.process_cpu_percent:.1f}%")
            logger.info(f"  프로세스 RAM: {snapshot.process_ram_gb:.2f}GB")
            logger.info(f"  프로세스 스레드: {snapshot.process_threads}")
        
        # 상위 프로세스
        top = self.get_top_processes(3)
        logger.info("  Top 3 프로세스:")
        for i, proc in enumerate(top, 1):
            logger.info(f"    {i}. {proc['name']} (PID: {proc['pid']}): CPU {proc['cpu_percent']:.1f}% | RAM {proc['memory_percent']:.1f}%")
        
        logger.info("="*80)
        
        return snapshot
    
    def check_memory_pressure(self, threshold: float = 90.0) -> bool:
        """메모리 압박 상태 체크"""
        ram = psutil.virtual_memory()
        return ram.percent >= threshold
    
    def get_summary(self) -> Dict:
        """모니터링 요약 통계"""
        if not self.snapshots:
            return {}
        
        cpu_percents = [s.cpu_percent for s in self.snapshots]
        ram_percents = [s.ram_percent for s in self.snapshots]
        
        return {
            'cpu_avg': sum(cpu_percents) / len(cpu_percents),
            'cpu_max': max(cpu_percents),
            'cpu_min': min(cpu_percents),
            'ram_avg': sum(ram_percents) / len(ram_percents),
            'ram_max': max(ram_percents),
            'ram_min': min(ram_percents),
            'snapshots_count': len(self.snapshots)
        }


if __name__ == "__main__":
    # 테스트
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    monitor = CPUMonitor()
    monitor.log_snapshot(logger, "테스트")
    
    print("\n상세 정보:")
    import json
    print(json.dumps(monitor.get_detailed_info(), indent=2))


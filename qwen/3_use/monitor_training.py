#!/usr/bin/env python3
"""
학습 모니터링 스크립트 (TensorBoard 대안)
포트 접근 불가 환경에서 사용
"""

import re
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def parse_log_file(log_path):
    """로그 파일에서 메트릭 추출"""
    metrics = {
        'steps': [],
        'train_loss': [],
        'eval_loss': [],
        'learning_rate': [],
        'embedding_similarity': [],
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # Step 추출
            step_match = re.search(r"'step': (\d+)", line)
            if step_match:
                current_step = int(step_match.group(1))
            
            # Train Loss
            loss_match = re.search(r"'loss': ([\d.]+)", line)
            if loss_match:
                metrics['steps'].append(current_step)
                metrics['train_loss'].append(float(loss_match.group(1)))
            
            # Eval Loss
            eval_match = re.search(r"'eval_loss': ([\d.]+)", line)
            if eval_match:
                metrics['eval_loss'].append(float(eval_match.group(1)))
            
            # Learning Rate
            lr_match = re.search(r"'learning_rate': ([\d.e-]+)", line)
            if lr_match:
                metrics['learning_rate'].append(float(lr_match.group(1)))
            
            # 임베딩 유사도
            emb_match = re.search(r"'데이터' ↔ '분석' 유사도: ([\d.]+)", line)
            if emb_match:
                metrics['embedding_similarity'].append(float(emb_match.group(1)))
    
    return metrics


def plot_metrics(metrics, output_dir):
    """메트릭을 그래프로 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss 그래프
    if metrics['train_loss']:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics['steps'], metrics['train_loss'], label='Train Loss', marker='o')
        if metrics['eval_loss']:
            eval_steps = metrics['steps'][::len(metrics['steps'])//len(metrics['eval_loss'])][:len(metrics['eval_loss'])]
            plt.plot(eval_steps, metrics['eval_loss'], label='Eval Loss', marker='s')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training & Evaluation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Learning Rate
        plt.subplot(1, 2, 2)
        if metrics['learning_rate']:
            plt.plot(metrics['steps'], metrics['learning_rate'], label='Learning Rate', color='green')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_and_lr.png', dpi=150)
        print(f"✓ 그래프 저장: {output_dir / 'loss_and_lr.png'}")
        plt.close()
    
    # 3. 임베딩 유사도
    if metrics['embedding_similarity']:
        plt.figure(figsize=(10, 6))
        steps_emb = list(range(500, 500 * (len(metrics['embedding_similarity']) + 1), 500))[:len(metrics['embedding_similarity'])]
        plt.plot(steps_emb, metrics['embedding_similarity'], marker='o', linewidth=2, markersize=8)
        plt.axhline(y=0.3, color='orange', linestyle='--', label='권장 (0.3)')
        plt.axhline(y=0.5, color='green', linestyle='--', label='우수 (0.5)')
        plt.axhline(y=0.7, color='blue', linestyle='--', label='매우 우수 (0.7)')
        plt.xlabel('Steps')
        plt.ylabel('Similarity')
        plt.title("'데이터' ↔ '분석' 임베딩 유사도")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'embedding_similarity.png', dpi=150)
        print(f"✓ 그래프 저장: {output_dir / 'embedding_similarity.png'}")
        plt.close()


def print_summary(metrics):
    """메트릭 요약 출력"""
    print("\n" + "="*80)
    print("학습 메트릭 요약")
    print("="*80)
    
    if metrics['train_loss']:
        print(f"\n[ Train Loss ]")
        print(f"  초기: {metrics['train_loss'][0]:.4f}")
        print(f"  최종: {metrics['train_loss'][-1]:.4f}")
        print(f"  감소: {metrics['train_loss'][0] - metrics['train_loss'][-1]:.4f}")
    
    if metrics['eval_loss']:
        print(f"\n[ Eval Loss ]")
        print(f"  최소: {min(metrics['eval_loss']):.4f}")
        print(f"  최종: {metrics['eval_loss'][-1]:.4f}")
    
    if metrics['embedding_similarity']:
        print(f"\n[ 임베딩 유사도 ]")
        print(f"  초기: {metrics['embedding_similarity'][0]:.4f}")
        print(f"  최종: {metrics['embedding_similarity'][-1]:.4f}")
        print(f"  증가: {metrics['embedding_similarity'][-1] - metrics['embedding_similarity'][0]:.4f}")
        
        final_sim = metrics['embedding_similarity'][-1]
        if final_sim > 0.7:
            status = "매우 우수 ✅"
        elif final_sim > 0.5:
            status = "우수 ✅"
        elif final_sim > 0.3:
            status = "권장 수준"
        else:
            status = "개선 필요 ⚠️"
        print(f"  평가: {status}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='학습 로그 모니터링')
    parser.add_argument('--log', type=str, required=True, help='로그 파일 경로')
    parser.add_argument('--output', type=str, default='./monitoring_output', help='그래프 출력 디렉토리')
    args = parser.parse_args()
    
    print(f"\n로그 파일 분석 중: {args.log}")
    
    # 로그 파싱
    metrics = parse_log_file(args.log)
    
    # 그래프 생성
    plot_metrics(metrics, args.output)
    
    # 요약 출력
    print_summary(metrics)
    
    print(f"\n✓ 완료! 그래프를 확인하세요: {args.output}/")


if __name__ == "__main__":
    main()


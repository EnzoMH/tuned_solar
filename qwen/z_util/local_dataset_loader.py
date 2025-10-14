#!/usr/bin/env python3
"""
로컬 JSON 데이터셋 로더
- 스트리밍 문제 발생 시 대안
- /home/work/tesseract/korean_large_data/korean_large_dataset.json 사용
"""

import json
from datasets import Dataset
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_local_dataset(
    file_path: str = "/home/work/tesseract/korean_large_data/korean_large_dataset.json",
    max_samples: Optional[int] = None
) -> Dataset:
    """
    로컬 JSON 파일에서 데이터셋 로드
    
    Args:
        file_path: JSON 파일 경로
        max_samples: 로드할 최대 샘플 수 (None이면 전체)
    
    Returns:
        Dataset: HuggingFace Dataset 객체
    """
    logger.info(f"로컬 데이터셋 로딩: {file_path}")
    
    try:
        # JSON 파일 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"  JSON 로드 완료: {len(data):,}개 샘플")
        
        # 샘플 수 제한
        if max_samples and len(data) > max_samples:
            logger.info(f"  샘플 제한: {len(data):,} → {max_samples:,}")
            data = data[:max_samples]
        
        # Dataset 변환
        dataset = Dataset.from_list(data)
        logger.info(f"[ COMPLETE ] 데이터셋 변환 완료: {len(dataset):,}개")
        
        # 데이터 형식 확인
        if len(dataset) > 0:
            first_sample = dataset[0]
            logger.info(f"  샘플 키: {list(first_sample.keys())}")
            
            # messages 필드 확인
            if 'messages' in first_sample:
                logger.info(f"  ✓ 'messages' 필드 확인")
            else:
                logger.warning(f"  [ WARNING ]  'messages' 필드 없음! 데이터 형식 확인 필요")
        
        return dataset
        
    except FileNotFoundError:
        logger.error(f"[ FAIL ] 파일을 찾을 수 없습니다: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[ FAIL ] JSON 파싱 오류: {e}")
        raise
    except Exception as e:
        logger.error(f"[ FAIL ] 데이터셋 로딩 실패: {e}")
        raise


def convert_to_messages_format(dataset: Dataset, 
                                instruction_key: str = "instruction",
                                output_key: str = "output") -> Dataset:
    """
    데이터셋을 messages 형식으로 변환
    
    Args:
        dataset: 원본 데이터셋
        instruction_key: 질문/지시 필드명
        output_key: 답변 필드명
    
    Returns:
        Dataset: messages 형식으로 변환된 데이터셋
    """
    logger.info("데이터셋을 messages 형식으로 변환 중...")
    
    def format_to_messages(example):
        """단일 샘플을 messages 형식으로 변환"""
        # 이미 messages 형식이면 그대로 반환
        if 'messages' in example:
            return example
        
        # messages 형식으로 변환
        messages = []
        
        # System 메시지 (선택적)
        if 'system' in example:
            messages.append({
                "role": "system",
                "content": example['system']
            })
        
        # User 메시지
        if instruction_key in example:
            messages.append({
                "role": "user",
                "content": example[instruction_key]
            })
        
        # Assistant 메시지
        if output_key in example:
            messages.append({
                "role": "assistant",
                "content": example[output_key]
            })
        
        return {"messages": messages}
    
    converted = dataset.map(format_to_messages, remove_columns=dataset.column_names)
    logger.info(f"[ COMPLETE ] 변환 완료: {len(converted):,}개")
    
    return converted


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    # 로컬 데이터셋 로드 테스트
    try:
        dataset = load_local_dataset(max_samples=10)
        print(f"\n샘플 확인:")
        print(dataset[0])
    except Exception as e:
        print(f"테스트 실패: {e}")


"""WMS Instruction Tuning Dataset Generator - 메인 실행 파일"""

from pathlib import Path
from config import config
from personas import WMS_TOPICS
from vectordb_loader import FAISSVectorDBLoader
from question_generator import GeminiQuestionGenerator
from answer_generator import SOLARAnswerGenerator
from dataset_builder import QADatasetBuilder


def main():
    """메인 파이프라인"""
    
    print("\n" + "=" * 80)
    print(" WMS Instruction Tuning Dataset Generator")
    print(" Role-based Approach")
    print("=" * 80)
    print("\n역할 분담:")
    print("  • Gemini 1.5 Flash = 물류 현업자 (질문 생성)")
    print("  • SOLAR + FAISS = WMS 전문가 (답변 생성)")
    print("=" * 80)
    
    # Step 1: 설정 검증
    print("\n### 설정 검증")
    try:
        config.validate()
        print("✓ 설정 검증 완료")
    except Exception as e:
        print(f"❌ 설정 오류: {e}")
        return
    
    # Step 2: 벡터 DB 로딩
    print("\n### FAISS 벡터 DB 로딩")
    
    try:
        vectordb = FAISSVectorDBLoader()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n다음 단계:")
        print("1. 기존 faiss_storage 디렉토리를 Instruction/data/vectordb/로 복사")
        print("2. 다시 실행")
        return
    
    # Step 3: 컴포넌트 초기화
    print("\n### 컴포넌트 초기화")
    
    try:
        # Gemini (물류 현업자)
        question_gen = GeminiQuestionGenerator()
        
        # SOLAR (WMS 전문가 + VectorDB)
        answer_gen = SOLARAnswerGenerator(vectordb_loader=vectordb)
        
    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        return
    
    # Step 4: Dataset Builder
    print("\n### 데이터셋 빌더 초기화")
    builder = QADatasetBuilder(question_gen, answer_gen)
    
    # Step 5: 주제 출력
    print(f"\n### 생성할 주제: {len(WMS_TOPICS)}개")
    for i, topic in enumerate(WMS_TOPICS, 1):
        print(f"  {i}. {topic}")
    
    # Step 6: 데이터셋 생성
    print(f"\n### 데이터셋 생성 시작")
    print(f"  페르소나당 질문 수: {config.QUESTIONS_PER_TOPIC}개")
    print(f"  예상 총 QA 수: {len(WMS_TOPICS)} × 5명 × {config.QUESTIONS_PER_TOPIC} = {len(WMS_TOPICS) * 5 * config.QUESTIONS_PER_TOPIC}개")
    
    dataset = builder.build_dataset(
        topics=WMS_TOPICS,
        questions_per_topic=config.QUESTIONS_PER_TOPIC
    )
    
    # Step 7: 저장
    output_path = builder.save_dataset(dataset)
    
    print("\n" + "=" * 80)
    print(" ✅ 완료!")
    print("=" * 80)
    print(f"\n출력 파일: {output_path}")
    print(f"다음 단계: 이 파일로 SOLAR 파인튜닝 진행")


if __name__ == "__main__":
    main()


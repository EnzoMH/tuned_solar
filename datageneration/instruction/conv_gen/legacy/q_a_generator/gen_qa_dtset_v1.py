"""
WMS QA Dataset 생성 스크립트
conv_gen 모듈을 사용한 간단한 QA 데이터셋 생성
"""

import sys
from pathlib import Path

# conv_gen 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

# EXAONE 모드 사용 시 동적 import
def get_question_maker():
    """EXAONE Question Maker를 동적으로 import"""
    from conv_gen.q_gen import ExaoneQuestionMaker, WMS_TOPICS
    from conv_gen.personas import PERSONAS
    return ExaoneQuestionMaker, WMS_TOPICS, PERSONAS

def get_answer_maker(use_vllm: bool = False):
    """Answer Maker를 선택적으로 import"""
    if use_vllm:
        from conv_gen.a_maker_vl import AnswerMakerVLLM
        return AnswerMakerVLLM()
    else:
        from conv_gen.a_maker_2 import AnswerMakerV2
        return AnswerMakerV2()


def generate_with_exaone(
    num_topics: int = 2,
    num_personas: int = 2,
    questions_per_topic: int = 2,
    output_dir: str = "output",
    use_vllm: bool = False
):
    """
    방법 1: EXAONE으로 질문 생성 + EXAONE으로 답변 생성 (RAG)
    """
    engine = "vLLM" if use_vllm else "Transformers"
    print("\n" + "="*80)
    print(f" QA Dataset 생성 시작 (EXAONE Q + EXAONE A + RAG) - Engine: {engine}")
    print("="*80 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # STEP 1: EXAONE으로 질문 생성
    print("\n[1/3] EXAONE 질문 생성 중...")
    
    # 동적 import
    ExaoneQuestionMaker, WMS_TOPICS, PERSONAS = get_question_maker()
    q_gen = ExaoneQuestionMaker()
    
    questions_data = q_gen.generate_diverse_questions(
        topics=WMS_TOPICS[:num_topics],
        personas=PERSONAS[:num_personas],
        questions_per_topic=questions_per_topic
    )
    
    # 질문만 추출
    questions = [q['question'] for q in questions_data]
    print(f"✓ {len(questions)}개 질문 생성 완료")
    
    # Question Maker 메모리 정리
    q_gen.cleanup()
    
    # STEP 2: EXAONE으로 답변 생성
    print(f"\n[2/3] EXAONE 답변 생성 중 (FAISS RAG + {engine})...")
    a_maker = get_answer_maker(use_vllm=use_vllm)
    
    answers = a_maker.generate_answers_batch(questions)
    print(f"✓ {len(answers)}개 답변 생성 완료")
    
    # STEP 3: Q-A 페어 저장
    print("\n[3/3] 데이터셋 저장 중...")
    
    import json
    from datetime import datetime
    
    qa_dataset = []
    for i, (q_data, a_data) in enumerate(zip(questions_data, answers), 1):
        # LLaMA 3.1/3.2 Instruction Tuning 형식
        qa_dataset.append({
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an AI assistant specialized in Warehouse Management Systems (WMS) and logistics automation. Provide clear, professional, and actionable advice based on real-world implementation experiences.'
                },
                {
                    'role': 'user',
                    'content': q_data['question']
                },
                {
                    'role': 'assistant',
                    'content': a_data['answer']
                }
            ],
            'metadata': {
                'id': i,
                'topic': q_data['topic'],
                'persona': q_data['persona'],
                'num_contexts': a_data['num_contexts'],
                'inference_time': a_data['inference_time_sec'],
                'output_tokens': a_data['output_tokens']
            }
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"wms_qa_dataset_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 저장 완료: {output_file}")
    
    # 정리
    a_maker.cleanup()
    
    print("\n" + "="*80)
    print(f" 완료! 총 {len(qa_dataset)}개의 Q-A 페어 생성")
    print("="*80 + "\n")
    
    return str(output_file)


def generate_with_manual_questions(
    questions: list,
    output_dir: str = "output",
    use_vllm: bool = False
):
    """
    방법 2: 수동 질문 리스트 + EXAONE으로 답변 생성 (RAG)
    """
    engine = "vLLM" if use_vllm else "Transformers"
    print("\n" + "="*80)
    print(f" QA Dataset 생성 시작 (수동 질문 + EXAONE A + RAG) - Engine: {engine}")
    print("="*80 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # STEP 1: EXAONE으로 답변 생성
    print(f"\n[1/2] EXAONE 답변 생성 중... (질문 {len(questions)}개, {engine})")
    a_maker = get_answer_maker(use_vllm=use_vllm)
    
    answers = a_maker.generate_answers_batch(questions)
    print(f"✓ {len(answers)}개 답변 생성 완료")
    
    # STEP 2: Q-A 페어 저장
    print("\n[2/2] 데이터셋 저장 중...")
    
    import json
    from datetime import datetime
    
    qa_dataset = []
    for i, a_data in enumerate(answers, 1):
        # LLaMA 3.1/3.2 Instruction Tuning 형식
        qa_dataset.append({
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an AI assistant specialized in Warehouse Management Systems (WMS) and logistics automation. Provide clear, professional, and actionable advice based on real-world implementation experiences.'
                },
                {
                    'role': 'user',
                    'content': a_data['question']
                },
                {
                    'role': 'assistant',
                    'content': a_data['answer']
                }
            ],
            'metadata': {
                'id': i,
                'num_contexts': a_data['num_contexts'],
                'inference_time': a_data['inference_time_sec'],
                'output_tokens': a_data['output_tokens']
            }
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"wms_qa_dataset_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 저장 완료: {output_file}")
    
    # 정리
    a_maker.cleanup()
    
    print("\n" + "="*80)
    print(f" 완료! 총 {len(qa_dataset)}개의 Q-A 페어 생성")
    print("="*80 + "\n")
    
    return str(output_file)


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WMS QA Dataset 생성')
    parser.add_argument(
        '--mode',
        choices=['exaone', 'manual'],
        default='manual',
        help='생성 모드: exaone (EXAONE 질문 생성) 또는 manual (수동 질문)'
    )
    parser.add_argument(
        '--num-topics',
        type=int,
        default=5,
        help='EXAONE 모드에서 사용할 주제 수 (기본: 5)'
    )
    parser.add_argument(
        '--num-personas',
        type=int,
        default=5,
        help='EXAONE 모드에서 사용할 페르소나 수 (기본: 5)'
    )
    parser.add_argument(
        '--questions-per-topic',
        type=int,
        default=1,
        help='EXAONE 모드에서 주제당 질문 수 (기본: 1)'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='출력 디렉토리 (기본: output)'
    )
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        help='vLLM 엔진 사용 (H100 2개 활용, 10배 이상 속도 향상)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'exaone':
        # EXAONE으로 질문 생성 + EXAONE으로 답변
        generate_with_exaone(
            num_topics=args.num_topics,
            num_personas=args.num_personas,
            questions_per_topic=args.questions_per_topic,
            output_dir=args.output_dir,
            use_vllm=args.use_vllm
        )
    else:
        # 수동 질문 리스트 (테스트용 5개)
        manual_questions = [
            "WMS 시스템 도입 시 초기 투자 비용은 대략 어느 정도야?",
            "재고 실사 시간을 현재 8시간에서 2-3시간으로 줄일 수 있는 방법이 있어?",
            "IT 경험이 없는 50대 직원들도 쉽게 사용할 수 있는 WMS가 있어?",
            "엑셀로 관리하던 3년치 재고 데이터를 새 시스템으로 옮기는 게 가능해?",
            "피크 시즌에 주문량이 평소보다 3배 늘어나도 시스템이 안정적으로 작동해?",
        ]
        
        print(f"\n사용할 질문 ({len(manual_questions)}개):")
        for i, q in enumerate(manual_questions, 1):
            print(f"  {i}. {q}")
        print()
        
        generate_with_manual_questions(
            questions=manual_questions,
            output_dir=args.output_dir,
            use_vllm=args.use_vllm
        )


if __name__ == "__main__":
    main()


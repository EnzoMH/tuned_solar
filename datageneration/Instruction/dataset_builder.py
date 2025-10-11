"""QA 데이터셋 빌더 (Gemini + SOLAR)"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from question_generator import GeminiQuestionGenerator
from answer_generator import SOLARAnswerGenerator
from config import config


class QADatasetBuilder:
    """Gemini(질문) + SOLAR(답변) 데이터셋 생성"""
    
    def __init__(
        self,
        question_generator: GeminiQuestionGenerator,
        answer_generator: SOLARAnswerGenerator
    ):
        self.q_gen = question_generator
        self.a_gen = answer_generator
    
    def build_dataset(
        self,
        topics: List[str],
        questions_per_topic: int = None
    ) -> List[Dict]:
        """전체 데이터셋 생성"""
        
        if questions_per_topic is None:
            questions_per_topic = config.QUESTIONS_PER_TOPIC
        
        print("\n" + "=" * 80)
        print(" QA 데이터셋 생성 파이프라인")
        print("=" * 80)
        
        # Step 1: Gemini로 질문 생성
        print("\n### STEP 1: Gemini - 물류 현업자 질문 생성")
        question_data = self.q_gen.generate_diverse_questions(
            topics=topics,
            questions_per_topic=questions_per_topic
        )
        
        # Step 2: SOLAR로 답변 생성
        print("\n### STEP 2: SOLAR - WMS 전문가 답변 생성")
        print("=" * 80)
        
        dataset = []
        
        for idx, q_data in enumerate(question_data, 1):
            question = q_data["question"]
            persona_name = q_data["persona"]
            persona_background = q_data["background"]
            
            print(f"\n[{idx}/{len(question_data)}] 답변 생성 중...")
            print(f"  페르소나: {persona_name}")
            print(f"  Q: {question}")
            
            # SOLAR 답변 생성 (Persona 정보 전달)
            answer_result = self.a_gen.generate_answer(
                question=question,
                persona_name=persona_name,
                persona_background=persona_background
            )
            
            print(f"  A: {answer_result['answer']}")
            print(f"  검색 문서: {answer_result['num_docs_retrieved']}개")
            print("-" * 80)
            
            # Instruction format
            qa_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer_result['answer']
                    }
                ],
                "metadata": {
                    "topic": q_data["topic"],
                    "persona": q_data["persona"],
                    "persona_background": q_data["background"],
                    "context_used": answer_result['context_used'],
                    "num_docs_retrieved": answer_result['num_docs_retrieved'],
                    "created_at": datetime.now().isoformat()
                }
            }
            
            dataset.append(qa_item)
        
        print("\n" + "=" * 80)
        print(f" 데이터셋 생성 완료: 총 {len(dataset)}개")
        print("=" * 80)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "wms_instruction_dataset.json"):
        """데이터셋 저장"""
        
        output_path = config.OUTPUT_DIR / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 데이터셋 저장: {output_path}")
        
        # 통계
        self._print_statistics(dataset)
        
        return output_path
    
    def _print_statistics(self, dataset: List[Dict]):
        """데이터셋 통계 출력"""
        print(f"\n📊 데이터셋 통계:")
        print(f"  총 샘플 수: {len(dataset)}")
        
        # 페르소나별 통계
        personas = {}
        topics = {}
        for item in dataset:
            p = item["metadata"]["persona"]
            t = item["metadata"]["topic"]
            personas[p] = personas.get(p, 0) + 1
            topics[t] = topics.get(t, 0) + 1
        
        print(f"\n  페르소나별:")
        for p, count in personas.items():
            print(f"    • {p}: {count}개")
        
        print(f"\n  주제별:")
        for t, count in topics.items():
            print(f"    • {t}: {count}개")
        
        # 샘플 출력
        print(f"\n📝 샘플 QA:")
        if dataset:
            sample = dataset[0]
            print(f"  페르소나: {sample['metadata']['persona']}")
            print(f"  Q: {sample['messages'][0]['content']}")
            print(f"  A: {sample['messages'][1]['content'][:150]}...")



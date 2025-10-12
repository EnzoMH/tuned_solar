"""
WMS QA Dataset V2 - 대규모 생성 (50K samples)
- expanded_data/ 디렉토리에서 personas & topics 동적 로드
- 배치 처리 + 체크포인트
- vLLM 기반 고속 생성
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import gc
import torch

# conv_gen 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))


class QADatasetGeneratorV2:
    """대규모 QA 데이터셋 생성기"""
    
    def __init__(
        self,
        expanded_data_dir: str = "expanded_data",
        output_dir: str = "output",
        use_vllm: bool = True
    ):
        self.expanded_data_dir = Path(expanded_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.use_vllm = use_vllm
        
        print("\n" + "="*80)
        print("QA Dataset Generator V2 초기화")
        print(f"Mode: {'vLLM (High-Speed)' if use_vllm else 'Transformers'}")
        print("="*80 + "\n")
        
        # Personas & Topics 로드
        self.personas = self._load_personas()
        self.topics_technical = self._load_topics("technical")
        self.topics_practical = self._load_topics("practical")
        self.all_topics = self.topics_technical + self.topics_practical
        
        print(f"✓ Personas 로드: {len(self.personas)}개")
        print(f"✓ Technical Topics: {len(self.topics_technical)}개")
        print(f"✓ Practical Topics: {len(self.topics_practical)}개")
        print(f"✓ 총 Topics: {len(self.all_topics)}개\n")
        
        # Answer Maker 초기화 (vLLM)
        if self.use_vllm:
            from conv_gen.a_maker_vl import AnswerMakerVLLM
            self.answer_maker = AnswerMakerVLLM()
        else:
            raise NotImplementedError("V2는 vLLM 전용입니다. use_vllm=True로 설정하세요.")
    
    def _load_personas(self) -> List[Dict]:
        """Personas 로드 (여러 파일 병합)"""
        personas = []
        persona_files = sorted(self.expanded_data_dir.glob("personas_*.json"))
        
        for file in persona_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 리스트 형태로 직접 저장되어 있음
                if isinstance(data, list):
                    personas.extend(data)
                else:
                    # 혹시 다른 형식이면
                    personas.append(data)
        
        return personas
    
    def _load_topics(self, topic_type: str) -> List[str]:
        """Topics 로드 (technical 또는 practical)"""
        topics = []
        topic_files = sorted(self.expanded_data_dir.glob(f"topics_*_{topic_type}.json"))
        
        for file in topic_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # topics 키로 저장되어 있음
                if 'topics' in data:
                    topics.extend(data['topics'])
                elif isinstance(data, list):
                    topics.extend(data)
        
        return topics
    
    def generate_questions_for_batch(
        self,
        batch_personas: List[Dict],
        batch_topics: List[str],
        questions_per_combo: int = 1
    ) -> List[Dict]:
        """배치 단위로 질문 생성 (EXAONE)"""
        from conv_gen.q_gen import ExaoneQuestionMaker
        
        # Question Maker 초기화
        q_gen = ExaoneQuestionMaker()
        
        questions_data = []
        
        # 각 페르소나 × 토픽 조합별로 질문 생성
        for persona in batch_personas:
            for topic in batch_topics:
                # EXAONE으로 질문 생성
                questions = q_gen.generate_diverse_questions(
                    topics=[topic],
                    personas=[persona],
                    questions_per_topic=questions_per_combo
                )
                questions_data.extend(questions)
        
        # Question Maker 정리
        q_gen.cleanup()
        del q_gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return questions_data
    
    def generate_batch(
        self,
        batch_personas: List[Dict],
        batch_topics: List[str],
        questions_per_combo: int = 1,
        batch_id: int = 0
    ) -> List[Dict]:
        """배치 단위 QA 생성"""
        print(f"\n{'='*80}")
        print(f"Batch {batch_id}: {len(batch_personas)} personas × {len(batch_topics)} topics")
        print(f"{'='*80}\n")
        
        # STEP 1: 질문 생성
        print(f"[1/2] 질문 생성 중...")
        questions_data = self.generate_questions_for_batch(
            batch_personas,
            batch_topics,
            questions_per_combo
        )
        questions = [q['question'] for q in questions_data]
        print(f"✓ {len(questions)}개 질문 생성 완료\n")
        
        # STEP 2: 답변 생성 (vLLM)
        print(f"[2/2] 답변 생성 중 (vLLM + FAISS RAG)...")
        answers = self.answer_maker.generate_answers_batch(questions)
        print(f"✓ {len(answers)}개 답변 생성 완료\n")
        
        # STEP 3: QA 페어 조합
        qa_dataset = []
        for i, (q_data, a_data) in enumerate(zip(questions_data, answers), 1):
            qa_dataset.append({
                'messages': [
                    {
                        'role': 'system',
                        'content': '당신은 10년 경력의 물류 시스템 전문가입니다. WMS 도입과 운영에 대해 실무 경험을 바탕으로 구체적이고 실용적인 조언을 제공하세요.'
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
                    'batch_id': batch_id,
                    'topic': q_data.get('topic', 'unknown'),
                    'persona': q_data.get('persona', {}).get('name', 'unknown'),
                    'question_style': q_data.get('persona', {}).get('question_style', ''),
                    'inference_time': a_data.get('inference_time', 0),
                    'context_used': len(a_data.get('contexts', []))
                }
            })
        
        return qa_dataset
    
    def generate_large_dataset(
        self,
        total_samples: int = 50000,
        batch_size: int = 100,
        checkpoint_every: int = 1000,
        questions_per_combo: int = 1
    ):
        """대규모 QA 데이터셋 생성"""
        print("\n" + "="*80)
        print(f"대규모 QA 데이터셋 생성 시작")
        print(f"목표: {total_samples:,} 샘플")
        print(f"배치 크기: {batch_size}")
        print(f"체크포인트: 매 {checkpoint_every} 샘플")
        print("="*80 + "\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_qa_data = []
        batch_count = 0
        
        # 총 필요한 조합 수 계산
        needed_combos = total_samples // questions_per_combo
        
        import itertools
        import random
        
        # Persona × Topic 조합 생성
        all_combos = list(itertools.product(self.personas, self.all_topics))
        random.shuffle(all_combos)  # 랜덤화
        
        print(f"✓ 가능한 조합 수: {len(all_combos):,}개")
        print(f"✓ 필요한 조합 수: {needed_combos:,}개\n")
        
        # 조합이 부족하면 반복
        if len(all_combos) < needed_combos:
            repeat_times = (needed_combos // len(all_combos)) + 1
            all_combos = all_combos * repeat_times
            random.shuffle(all_combos)
            print(f"✓ 조합 반복 ({repeat_times}회) → {len(all_combos):,}개\n")
        
        # 배치 단위로 생성
        for i in range(0, needed_combos, batch_size):
            batch_combos = all_combos[i:i+batch_size]
            batch_personas = [combo[0] for combo in batch_combos]
            batch_topics = [combo[1] for combo in batch_combos]
            
            batch_count += 1
            
            # 배치 생성
            batch_qa = self.generate_batch(
                batch_personas,
                batch_topics,
                questions_per_combo,
                batch_id=batch_count
            )
            
            all_qa_data.extend(batch_qa)
            
            print(f"✓ 현재까지 생성: {len(all_qa_data):,} / {total_samples:,} 샘플")
            print(f"  진행률: {len(all_qa_data)/total_samples*100:.1f}%\n")
            
            # 체크포인트 저장
            if len(all_qa_data) % checkpoint_every < batch_size:
                checkpoint_file = self.output_dir / f"checkpoint_{len(all_qa_data)}_{timestamp}.json"
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(all_qa_data, f, ensure_ascii=False, indent=2)
                print(f"💾 체크포인트 저장: {checkpoint_file}\n")
            
            # 목표 달성 시 종료
            if len(all_qa_data) >= total_samples:
                break
        
        # 최종 저장
        final_file = self.output_dir / f"wms_qa_dataset_v2_{total_samples}_{timestamp}.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_data[:total_samples], f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print(f"✓ 생성 완료: {len(all_qa_data[:total_samples]):,} 샘플")
        print(f"✓ 저장 위치: {final_file}")
        print("="*80 + "\n")
        
        # 통계 출력
        self._print_statistics(all_qa_data[:total_samples])
        
        return all_qa_data[:total_samples]
    
    def _print_statistics(self, qa_data: List[Dict]):
        """생성된 데이터 통계"""
        print("\n📊 데이터셋 통계:")
        print(f"  총 샘플 수: {len(qa_data):,}")
        
        # 평균 inference time
        total_time = sum(qa['metadata']['inference_time'] for qa in qa_data)
        avg_time = total_time / len(qa_data) if qa_data else 0
        print(f"  평균 Inference Time: {avg_time:.2f}초")
        print(f"  총 생성 시간: {total_time/60:.1f}분")
        
        # 토픽 분포
        from collections import Counter
        topics = [qa['metadata']['topic'] for qa in qa_data]
        top_topics = Counter(topics).most_common(5)
        print(f"\n  상위 5 토픽:")
        for topic, count in top_topics:
            print(f"    - {topic[:50]}...: {count}개")
        
        # 평균 답변 길이
        answer_lengths = [len(qa['messages'][2]['content']) for qa in qa_data]
        avg_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        print(f"\n  평균 답변 길이: {avg_length:.0f} 자")


def main():
    parser = argparse.ArgumentParser(description='WMS QA Dataset V2 Generator')
    parser.add_argument('--expanded-data-dir', type=str, default='expanded_data',
                        help='Personas & Topics 디렉토리')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='출력 디렉토리')
    parser.add_argument('--total-samples', type=int, default=50000,
                        help='생성할 총 샘플 수')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='배치당 생성 개수')
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                        help='체크포인트 저장 주기')
    parser.add_argument('--questions-per-combo', type=int, default=1,
                        help='Persona×Topic 조합당 질문 수')
    
    args = parser.parse_args()
    
    # Generator 초기화
    generator = QADatasetGeneratorV2(
        expanded_data_dir=args.expanded_data_dir,
        output_dir=args.output_dir,
        use_vllm=True
    )
    
    # 대규모 생성 실행
    generator.generate_large_dataset(
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        questions_per_combo=args.questions_per_combo
    )


if __name__ == "__main__":
    main()


"""
Q-A Dataset Generation Pipeline
EXAONE (Question Maker) + EEVE (Answer Maker) 통합 파이프라인
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from q_maker_exaone import ExaoneQuestionMaker  # EXAONE 질문 생성기
# from q_gen import GeminiQuestionGenerator  # Gemini 질문 생성기 (선택)
from a_maker import AnswerMaker
from personas import PERSONAS, WMS_TOPICS


class QAPipeline:
    def __init__(
        self,
        question_model: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        answer_model: str = "MyeongHo0621/eeve-vss-smh",
        faiss_path: str = "/home/work/tesseract/faiss_storage"
    ):
        """Q-A 파이프라인 초기화
        
        - EXAONE (질문): FAISS 미사용 (자체 지식으로 얕은 사전조사를 한 실무자)
        - EEVE (답변): FAISS 사용 (깊은 전문 지식의 10년 경력 전문가)
        """
        print(f"\n{'#'*80}", flush=True)
        print(f"Q-A Dataset Generation Pipeline", flush=True)
        print(f"{'#'*80}", flush=True)
        print(f"Question Model: {question_model} (FAISS 미사용)", flush=True)
        print(f"Answer Model: {answer_model} (FAISS 사용)", flush=True)
        print(f"FAISS Path (Answer only): {faiss_path}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        
        self.question_maker = None
        self.answer_maker = None
        self.question_model = question_model
        self.answer_model = answer_model
        self.faiss_path = faiss_path
    
    def generate_full_dataset(
        self,
        topics: Optional[List[str]] = None,
        personas: Optional[List[Dict]] = None,
        questions_per_topic: int = 3,
        output_dir: str = "/home/work/tesseract/datageneration/Instruction/output"
    ) -> Dict:
        """EXAONE 질문 생성 + EEVE 답변 생성 (전체 파이프라인)"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if topics is None:
            topics = WMS_TOPICS[:3]  # 기본: 처음 3개 주제
        if personas is None:
            personas = PERSONAS[:2]  # 기본: 처음 2개 페르소나
        
        # STEP 1: EXAONE으로 질문 생성 (FAISS 미사용)
        print(f"\n{'='*80}", flush=True)
        print(f"STEP 1: Question Generation (EXAONE, FAISS 미사용)", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        self.question_maker = ExaoneQuestionMaker(
            model_name=self.question_model
        )
        
        all_questions = self.question_maker.generate_diverse_questions(
            topics=topics,
            personas=personas,
            questions_per_topic=questions_per_topic
        )
        
        # 질문 저장
        questions_file = output_path / f"questions_{timestamp}.json"
        self.question_maker.save_questions(all_questions, str(questions_file))
        
        # Question Maker 메모리 정리
        self.question_maker.cleanup()
        
        # STEP 2: EEVE로 답변 생성 (FAISS 사용)
        print(f"\n{'='*80}", flush=True)
        print(f"STEP 2: Answer Generation (EEVE + FAISS RAG)", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        self.answer_maker = AnswerMaker(
            model_name=self.answer_model,
            faiss_path=self.faiss_path
        )
        
        # 질문 리스트 추출
        questions_list = [q['question'] for q in all_questions]
        
        # 답변 생성 (항상 FAISS RAG)
        answers = self.answer_maker.generate_answers_batch(
            questions=questions_list
        )
        
        # STEP 3: Q-A 페어 결합
        qa_dataset = []
        for i, (q_data, a_data) in enumerate(zip(all_questions, answers)):
            qa_pair = {
                'id': i + 1,
                'question': q_data['question'],
                'answer': a_data['answer'],
                'metadata': {
                    'topic': q_data['topic'],
                    'persona': q_data['persona'],
                    'question_model': self.question_model,
                    'answer_model': self.answer_model,
                    'question_inference_time': q_data['inference_time_sec'],
                    'answer_inference_time': a_data['inference_time_sec'],
                    'answer_input_tokens': a_data['input_tokens'],
                    'answer_output_tokens': a_data['output_tokens'],
                    'num_contexts': a_data['num_contexts'],
                    'timestamp': a_data['timestamp']
                }
            }
            qa_dataset.append(qa_pair)
        
        # 최종 데이터셋 저장
        dataset_file = output_path / f"qa_dataset_{timestamp}.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}", flush=True)
        print(f"Dataset Generation Complete!", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Questions file: {questions_file}", flush=True)
        print(f"Dataset file: {dataset_file}", flush=True)
        print(f"Total Q-A pairs: {len(qa_dataset)}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # 통계 생성
        stats = self._generate_statistics(qa_dataset)
        stats_file = output_path / f"statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Statistics saved to: {stats_file}\n", flush=True)
        
        # Answer Maker 메모리 정리
        self.answer_maker.cleanup()
        
        return {
            'dataset_file': str(dataset_file),
            'questions_file': str(questions_file),
            'stats_file': str(stats_file),
            'total_pairs': len(qa_dataset),
            'statistics': stats
        }
    
    def generate_answers_from_questions(
        self,
        questions: List[str],
        output_dir: str = "/home/work/tesseract/datageneration/Instruction/output"
    ) -> Dict:
        """질문 리스트로부터 답변 생성 (질문은 외부에서 제공)"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*80}", flush=True)
        print(f"질문 입력 받음: {len(questions)}개", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        print(f"\n{'='*80}", flush=True)
        print(f"STEP: Answer Generation", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # Answer Maker로 답변 생성
        self.answer_maker = AnswerMaker(
            model_name=self.answer_model,
            faiss_path=self.faiss_path
        )
        
        # 답변 생성 (항상 FAISS RAG)
        answers = self.answer_maker.generate_answers_batch(
            questions=questions
        )
        
        # Q-A 페어 생성
        qa_dataset = []
        for i, a_data in enumerate(answers):
            qa_pair = {
                'id': i + 1,
                'question': a_data['question'],
                'answer': a_data['answer'],
                'metadata': {
                    'answer_model': self.answer_model,
                    'answer_inference_time': a_data['inference_time_sec'],
                    'answer_input_tokens': a_data['input_tokens'],
                    'answer_output_tokens': a_data['output_tokens'],
                    'num_contexts': a_data['num_contexts'],
                    'timestamp': a_data['timestamp']
                }
            }
            qa_dataset.append(qa_pair)
        
        # 최종 데이터셋 저장
        dataset_file = output_path / f"qa_dataset_{timestamp}.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}", flush=True)
        print(f"Dataset Generation Complete!", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Dataset file: {dataset_file}", flush=True)
        print(f"Total Q-A pairs: {len(qa_dataset)}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # 통계 생성
        stats = self._generate_statistics(qa_dataset)
        stats_file = output_path / f"statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Statistics saved to: {stats_file}\n", flush=True)
        
        # Answer Maker 메모리 정리
        self.answer_maker.cleanup()
        
        return {
            'dataset_file': str(dataset_file),
            'stats_file': str(stats_file),
            'total_pairs': len(qa_dataset),
            'statistics': stats
        }
    
    def _generate_statistics(self, dataset: List[Dict]) -> Dict:
        """데이터셋 통계 생성"""
        total_questions = len(dataset)
        
        # 토큰 통계
        input_tokens = [item['metadata']['answer_input_tokens'] for item in dataset]
        output_tokens = [item['metadata']['answer_output_tokens'] for item in dataset]
        inference_times = [item['metadata']['answer_inference_time'] for item in dataset]
        
        # 주제별 통계 (있으면)
        topics = {}
        personas = {}
        for item in dataset:
            if 'topic' in item['metadata']:
                topic = item['metadata']['topic']
                topics[topic] = topics.get(topic, 0) + 1
            if 'persona' in item['metadata']:
                persona = item['metadata']['persona']
                personas[persona] = personas.get(persona, 0) + 1
        
        stats = {
            'total_qa_pairs': total_questions,
            'token_statistics': {
                'avg_input_tokens': sum(input_tokens) / len(input_tokens),
                'avg_output_tokens': sum(output_tokens) / len(output_tokens),
                'total_input_tokens': sum(input_tokens),
                'total_output_tokens': sum(output_tokens)
            },
            'performance': {
                'avg_inference_time_sec': sum(inference_times) / len(inference_times),
                'total_inference_time_sec': sum(inference_times),
                'avg_tokens_per_sec': sum(output_tokens) / sum(inference_times)
            },
            'models': {
                'question_model': self.question_model,
                'answer_model': self.answer_model
            }
        }
        
        if topics:
            stats['topics'] = topics
        if personas:
            stats['personas'] = personas
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """통계 출력"""
        print(f"\n{'='*80}", flush=True)
        print(f"Dataset Statistics", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        print(f"Total Q-A Pairs: {stats['total_qa_pairs']}", flush=True)
        
        # 주제별 통계 (있으면)
        if 'topics' in stats:
            print(f"\nTopics:", flush=True)
            for topic, count in stats['topics'].items():
                print(f"  - {topic}: {count} pairs", flush=True)
        
        # 페르소나별 통계 (있으면)
        if 'personas' in stats:
            print(f"\nPersonas:", flush=True)
            for persona, count in stats['personas'].items():
                print(f"  - {persona}: {count} pairs", flush=True)
        
        print(f"\nToken Statistics:", flush=True)
        print(f"  Average Input Tokens: {stats['token_statistics']['avg_input_tokens']:.1f}", flush=True)
        print(f"  Average Output Tokens: {stats['token_statistics']['avg_output_tokens']:.1f}", flush=True)
        print(f"  Total Tokens: {stats['token_statistics']['total_input_tokens'] + stats['token_statistics']['total_output_tokens']}", flush=True)
        
        print(f"\nPerformance:", flush=True)
        print(f"  Average Inference Time: {stats['performance']['avg_inference_time_sec']:.2f} sec", flush=True)
        print(f"  Total Inference Time: {stats['performance']['total_inference_time_sec']:.2f} sec", flush=True)
        print(f"  Average Tokens/sec: {stats['performance']['avg_tokens_per_sec']:.2f}", flush=True)
        
        print(f"\nModels:", flush=True)
        print(f"  Question: {stats['models']['question_model']}", flush=True)
        print(f"  Answer: {stats['models']['answer_model']}", flush=True)
        
        print(f"\n{'='*80}\n", flush=True)


def main():
    """파이프라인 실행 예시"""
    # 파이프라인 초기화
    pipeline = QAPipeline()
    
    # 방법 1: EXAONE 질문 생성 + EEVE 답변 생성 (전체 자동)
    print("\n방법 1: 전체 자동 파이프라인 (EXAONE + EEVE)")
    result = pipeline.generate_full_dataset(
        topics=WMS_TOPICS[:2],      # 처음 2개 주제
        personas=PERSONAS[:2],       # 처음 2개 페르소나
        questions_per_topic=2,       # 주제당 2개 질문
        output_dir="/home/work/tesseract/datageneration/Instruction/output"
    )
    
    # 통계 출력
    pipeline.print_statistics(result['statistics'])
    
    print(f"\n{'#'*80}", flush=True)
    print(f"Pipeline Complete!", flush=True)
    print(f"Dataset file: {result['dataset_file']}", flush=True)
    print(f"Questions file: {result['questions_file']}", flush=True)
    print(f"Total Q-A pairs: {result['total_pairs']}", flush=True)
    print(f"{'#'*80}\n", flush=True)
    
    # 방법 2: 질문만 직접 제공 (수동 질문 + EEVE 답변)
    # questions = [
    #     "WMS 도입 비용이 실제로 얼마나 드나요?",
    #     "재고 실사 시간을 얼마나 줄일 수 있나요?",
    # ]
    # result = pipeline.generate_answers_from_questions(questions)
    # pipeline.print_statistics(result['statistics'])


if __name__ == "__main__":
    main()


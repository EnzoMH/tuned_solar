"""
EXAONE 기반 질문 생성기
EXAONE 자체 지식만으로 WMS 관련 질문 생성 (FAISS 사용 안함)
얕은 사전조사를 한 실무자 페르소나
"""

import torch
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# personas 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "legacy" / "question"))
from personas import PERSONAS, WMS_TOPICS


class ExaoneQuestionMaker:
    """EXAONE 모델 기반 질문 생성기 (순수 LLM 지식 기반)"""
    
    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    ):
        """EXAONE 모델 초기화 (FAISS 없음)"""
        print(f"\n{'='*80}", flush=True)
        print(f"EXAONE Question Maker 초기화 (Model: {model_name})", flush=True)
        print(f"모드: 순수 LLM 지식 기반 (FAISS 미사용)", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # EXAONE 모델 로딩
        print(f"Loading EXAONE model...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print(f"✓ Model loaded: {self.model.device}", flush=True)
        print(f"✓ Memory: {self.model.get_memory_footprint() / 1024**3:.2f} GB\n", flush=True)
        
        self.model_name = model_name
    
    def create_question_prompt(
        self,
        topic: str,
        persona: Dict,
        num_questions: int = 5
    ) -> str:
        """AI Assistant에게 묻는 비즈니스 질문 생성 프롬프트"""
        
        prompt = f"""당신은 AI 전문가에게 WMS(창고관리시스템)에 대해 질문할 비즈니스 질문들을 작성하는 실무자입니다.

[질문자 배경]
- 이름: {persona['name']}
- 상황: {persona['background']}
- 주요 관심사: {', '.join(persona['concerns'])}
- 질문 스타일: {persona['question_style']}

주제: {topic}

위 배경을 가진 실무자가 AI Assistant에게 실제로 물어볼 만한 질문 {num_questions}개를 작성하세요.

질문 작성 원칙:
1. AI에게 조언을 구하는 형태 (예: "~알려주세요", "~어떤가요?", "~방법이 있나요?")
2. 구체적인 상황이나 숫자를 포함하면 더 좋음
3. 비즈니스 맥락이 명확해야 함
4. 한 문장으로 간결하게
5. 전문용어보다는 실무 언어 사용

좋은 질문 예시:
- "재고 실사 시간을 현재 8시간에서 2-3시간으로 줄일 수 있는 방법이 있어?"
- "IT 경험이 없는 50대 직원들도 쉽게 사용할 수 있는 WMS가 있어?"
- "엑셀로 관리하던 3년치 재고 데이터를 새 시스템으로 옮기는 게 가능해?"
- "초기 투자 비용을 단계적으로 나눠서 부담을 줄일 수 있나?"

나쁜 질문 예시:
- "WMS의 아키텍처 설계 방법론을 알려줘" (너무 기술적)
- "시스템이 좋은 시스템인가?" (모호함)
- "비용은 얼마야?" (불완전한 질문)

AI Assistant에게 물어볼 질문 {num_questions}개:
Q1:"""
        
        return prompt
    
    def generate_questions(
        self,
        topic: str,
        persona: Dict,
        num_questions: int = 5,
        max_new_tokens: int = 512
    ) -> List[Dict]:
        """주제와 페르소나에 맞는 질문 생성 (EXAONE 자체 지식만 사용)"""
        print(f"\n{'='*80}", flush=True)
        print(f"Generating questions", flush=True)
        print(f"Topic: {topic}", flush=True)
        print(f"Persona: {persona['name']}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # 프롬프트 생성 (FAISS 사용 안함)
        prompt = self.create_question_prompt(topic, persona, num_questions)
        
        # EXAONE chat template 적용
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 토크나이징
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 생성
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"Generated text:\n{generated_text}\n", flush=True)
        print(f"Inference time: {inference_time:.2f} sec\n", flush=True)
        
        # 질문 파싱
        questions = self._parse_questions(generated_text)
        
        # 결과 구조화
        results = []
        for q in questions[:num_questions]:
            results.append({
                'question': q,
                'topic': topic,
                'persona': persona['name'],
                'persona_background': persona['background'],
                'inference_time_sec': inference_time,
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    def _parse_questions(self, text: str) -> List[str]:
        """생성된 텍스트에서 질문 추출"""
        import re
        
        questions = []
        
        # 줄 단위로 분리
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 빈 줄이나 헤더 스킵
            if not line or line.startswith('#') or line.startswith('['):
                continue
            
            # 번호 제거 (Q1:, 1., 질문1:, - 등)
            line = re.sub(r'^[Qq][\d]+[\.\):\s]*', '', line)
            line = re.sub(r'^[\d]+[\.\):\s]+', '', line)
            line = re.sub(r'^질문[\d]+[\.\):\s]*', '', line)
            line = re.sub(r'^[-•\*]\s*', '', line)
            
            # 유효한 질문인지 확인 (길이, 물음표 포함 여부)
            if len(line) > 15 and ('?' in line or '요' in line or '까' in line):
                questions.append(line)
        
        return questions
    
    def generate_diverse_questions(
        self,
        topics: List[str],
        personas: Optional[List[Dict]] = None,
        questions_per_topic: int = 3
    ) -> List[Dict]:
        """다양한 주제와 페르소나로 질문 생성"""
        
        if personas is None:
            personas = PERSONAS
        
        print(f"\n{'#'*80}", flush=True)
        print(f"Diverse Question Generation", flush=True)
        print(f"Topics: {len(topics)}", flush=True)
        print(f"Personas: {len(personas)}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        
        all_questions = []
        
        for topic in topics:
            print(f"\n주제: {topic}")
            
            for persona in personas:
                print(f"  {persona['name']} 질문 생성 중...")
                
                questions = self.generate_questions(
                    topic=topic,
                    persona=persona,
                    num_questions=questions_per_topic
                )
                
                all_questions.extend(questions)
                print(f"    ✓ {len(questions)}개 생성")
        
        print(f"\n총 {len(all_questions)}개 질문 생성 완료!", flush=True)
        return all_questions
    
    def save_questions(self, questions: List[Dict], output_file: str):
        """질문 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        
        print(f"\nQuestions saved to: {output_file}", flush=True)
    
    def cleanup(self):
        """메모리 정리"""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """테스트 실행"""
    # Question Maker 초기화
    qm = ExaoneQuestionMaker()
    
    # 테스트: 처음 2개 주제, 2개 페르소나로 각 2개씩 질문 생성
    questions = qm.generate_diverse_questions(
        topics=WMS_TOPICS[:2],
        personas=PERSONAS[:2],
        questions_per_topic=2
    )
    
    # 저장
    qm.save_questions(questions, "generated_questions_exaone.json")
    
    # 생성된 질문 샘플 출력
    print("\n생성된 질문 샘플:")
    for q in questions[:5]:
        print(f"  [{q['persona']}] {q['question']}")
    
    # 정리
    qm.cleanup()


if __name__ == "__main__":
    main()

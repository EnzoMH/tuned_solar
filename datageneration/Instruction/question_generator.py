"""Gemini 기반 질문 생성기 (물류 현업자 역할)"""

import re
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from config import config
from personas import PERSONAS


class GeminiQuestionGenerator:
    """Gemini = 물류 현업자 (질문만 생성, VectorDB 사용 안함)"""
    
    def __init__(self):
        config.validate()
        
        print("\nGemini 2.0 Flash 초기화 (물류 현업자 역할)...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.8,
            max_output_tokens=800
        )
        print("✓ Gemini 초기화 완료")
    
    def generate_questions_with_persona(
        self,
        topic: str,
        persona: Dict,
        num_questions: int = 5
    ) -> List[str]:
        """페르소나 기반 질문 생성"""
        
        system_prompt = f"""당신은 {persona['name']}입니다.

[당신의 배경]
{persona['background']}

[당신의 주요 관심사]
{', '.join(persona['concerns'])}

[질문 스타일]
{persona['question_style']}

당신은 WMS(창고관리시스템) 도입을 진지하게 고민하고 있습니다.
실제 현업에서 궁금한 점들을 자연스럽게 질문하세요."""

        user_prompt = f"""주제: {topic}

이 주제에 대해 당신이 실제로 궁금해할 만한 질문 {num_questions}개를 생성하세요.

질문 작성 규칙:
1. 실무자 관점에서 실제로 물어볼 법한 자연스러운 질문
2. 너무 기술적이지 않고, 실용적이고 구체적인 질문
3. 당신의 관심사(비용, 효율, 직원 교육 등)가 반영된 질문
4. "WMS"라는 단어를 너무 반복하지 말고 자연스럽게
5. 각 질문은 한 문장으로

예시:
- "재고 실사할 때 시간이 너무 오래 걸리는데, 이걸 줄일 수 있나요?"
- "직원들이 시스템 사용법을 배우는데 얼마나 걸릴까요?"
- "기존에 쓰던 엑셀 데이터를 옮기는 게 가능한가요?"

질문 {num_questions}개:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # 응답 파싱
            questions = []
            for line in response.content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('['):
                    continue
                
                # 번호 제거
                line = re.sub(r'^[\d]+[\.\):\s]*', '', line)
                line = re.sub(r'^[Qq][\d]+[\.\):\s]*', '', line)
                line = re.sub(r'^질문[\d]+[\.\):\s]*', '', line)
                line = re.sub(r'^[-•\*]\s*', '', line)
                
                if line and len(line) > 10:
                    questions.append(line)
            
            return questions[:num_questions]
            
        except Exception as e:
            print(f"❌ 질문 생성 실패: {e}")
            return []
    
    def generate_diverse_questions(
        self,
        topics: List[str],
        questions_per_topic: int = 3
    ) -> List[Dict]:
        """다양한 페르소나로 질문 생성"""
        
        print("\n" + "=" * 80)
        print(" 물류 현업자 페르소나 기반 질문 생성")
        print("=" * 80)
        
        all_questions = []
        
        for topic in topics:
            print(f"\n주제: {topic}")
            
            for persona in PERSONAS:
                print(f"  {persona['name']} 질문 생성 중...")
                
                questions = self.generate_questions_with_persona(
                    topic=topic,
                    persona=persona,
                    num_questions=questions_per_topic
                )
                
                for q in questions:
                    all_questions.append({
                        "question": q,
                        "topic": topic,
                        "persona": persona['name'],
                        "background": persona['background']
                    })
                
                print(f"    ✓ {len(questions)}개 생성")
        
        print(f"\n총 {len(all_questions)}개 질문 생성 완료!")
        return all_questions


if __name__ == "__main__":
    from personas import WMS_TOPICS
    
    generator = GeminiQuestionGenerator()
    questions = generator.generate_diverse_questions(
        topics=WMS_TOPICS[:2],
        questions_per_topic=2
    )
    
    print("\n생성된 질문 샘플:")
    for q in questions[:5]:
        print(f"  [{q['persona']}] {q['question']}")


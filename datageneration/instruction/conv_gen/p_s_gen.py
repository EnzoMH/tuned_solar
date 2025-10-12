"""
Persona & Subject Generator (p_s_gen.py)
EXAONE vLLM으로 고품질 페르소나와 WMS/로봇자동화 토픽 자동 생성
"""

import json
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from vllm import LLM, SamplingParams


class PersonaTopicGenerator:
    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
        tensor_parallel_size: int = 2
    ):
        """EXAONE vLLM으로 페르소나/토픽 생성기 초기화"""
        print(f"\n{'='*80}")
        print(f"Persona & Topic Generator 초기화")
        print(f"Model: {model_name}")
        print(f"vLLM Tensor Parallelism: {tensor_parallel_size} GPUs")
        print(f"{'='*80}\n")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
            max_model_len=4096,
            trust_remote_code=True
        )
        
        self.model_name = model_name
        print("✓ 초기화 완료\n")
    
    def create_persona_generation_prompt(
        self,
        num_personas: int = 100,
        base_personas: List[Dict] = None
    ) -> str:
        """페르소나 생성 프롬프트"""
        
        base_examples = ""
        if base_personas:
            base_examples = "\n기존 페르소나 예시:\n"
            for p in base_personas[:3]:
                base_examples += f"- {p['name']}: {p['background']}, 관심사: {', '.join(p['concerns'])}\n"
        
        prompt = f"""당신은 물류 자동화 및 WMS 전문가입니다. 
다양한 산업군의 물류/창고 관련 실무자 페르소나 {num_personas}개를 생성하세요.

## 요구사항:
1. **다양한 업종**: 이커머스, 제조, 유통, 물류, 의료, 냉동/냉장, 식품, 패션, 가전, 화학, 자동차, 제약, 반도체, 건설자재 등
2. **다양한 직급**: CEO, CTO, 물류총괄, 운영팀장, 현장관리자, IT담당자, 물류컨설턴트, 구매팀, 총무팀, 전략기획팀, 회계팀 등
3. **다양한 규모**: 스타트업, 중소기업, 중견기업, 대기업, 글로벌 기업
4. **현실적인 배경**: 구체적인 문제 상황, 도입 동기
5. **실제 고민**: 기술적 고민, 비용 고민, 운영 고민을 구체적으로

{base_examples}

## 출력 형식 (JSON):
[
  {{
    "name": "홍길동 (직책 - 회사유형)",
    "background": "구체적인 배경 설명 (현재 시스템, 도입 동기)",
    "concerns": ["주요 고민1", "주요 고민2", "주요 고민3"],
    "question_style": "질문 스타일 설명"
  }},
  ...
]

**필수 준수사항**:
1. 반드시 유효한 JSON 배열만 출력 (주석 없음)
2. 모든 문자열은 큰따옴표(") 사용
3. question_style은 40자 이내로 간결하게
4. concerns는 정확히 3개 항목
5. background는 120자 이내

{num_personas}개의 페르소나 JSON (유효한 JSON 배열만):"""
        
        return prompt
    
    def create_topic_generation_prompt_technical(
        self,
        num_topics: int = 200,
        base_topics: List[str] = None
    ) -> str:
        """토픽 생성 프롬프트 - Technical (답변 품질 향상용)"""
        
        base_examples = ""
        if base_topics:
            base_examples = f"\n기존 토픽 예시:\n" + "\n".join([f"- {t}" for t in base_topics[:5]])
        
        prompt = f"""당신은 WMS 및 물류 로봇 자동화 전문가입니다.
고도로 전문적인 WMS/물류자동화 토픽 {num_topics}개를 생성하세요.

## 핵심 카테고리 및 비중:

### 1. 로봇 자동화 (40%) - 매우 세부적으로:
**AGV (Automated Guided Vehicle):**
- AGV 유형별 특징 (Wire-guided, Laser-guided, Vision-guided, Magnetic tape)
- AGV 최적 동선 계산 알고리즘 (Dijkstra, A*, Fleet management)
- AGV vs AMR (Autonomous Mobile Robot) 비교
- 멀티 AGV 충돌 방지 및 교통 관리
- AGV 배터리 관리 및 자동 충전 스테이션
- AGV 도입 ROI 계산 및 페이백 기간

**로봇 협업:**
- 사람+로봇 혼재 운영 (Collaborative workspace design)
- 안전 규정 (ISO 3691-4, ANSI/RIA R15.08)
- 로봇 피킹 시스템 (Piece picking, Case picking)
- 로보틱 암 vs 코봇 (Collaborative robot) 비교
- 인간-로봇 작업 분배 최적화

**스마트 팩토리 통합:**
- MES-WMS-ERP 3-way 연동
- IoT 센서 기반 실시간 모니터링
- Digital Twin 기술 활용
- 예측 정비 (Predictive maintenance)
- 5G 기반 원격 로봇 제어

**성과 측정:**
- MOP (Measure of Performance): 인간 vs 로봇 비교 지표
- KPI 설정: 처리량, 정확도, 가동률, MTBF, MTTR
- TCO (Total Cost of Ownership) 분석
- 에너지 효율성 측정

**기술 동향:**
- 현재 상용화 기술 (SOTA - State of the Art)
- 개발 중인 미래 기술 (Emerging technologies)
- AI/ML 기반 경로 최적화
- 컴퓨터 비전 기반 물체 인식
- 적용 가능성 및 기술 성숙도 (TRL - Technology Readiness Level)

### 2. WMS 운영 최적화 (30%):
- 피킹 전략 (Zone, Batch, Wave, Cluster picking)
- 재고 배치 최적화 (ABC 분석, Slotting optimization)
- 크로스 도킹 (Cross-docking) 전략
- 풀필먼트 센터 설계
- 리턴 프로세스 자동화

### 3. 시스템 통합 (15%):
- ERP-WMS-TMS 연동
- API 기반 통합 vs EDI
- 클라우드 WMS vs On-premise
- 마이크로서비스 아키텍처
- 실시간 데이터 동기화

### 4. 비용 및 ROI (10%):
- CAPEX vs OPEX 분석
- RaaS (Robot as a Service) 모델
- 리스 vs 구매 비교
- 인건비 절감 효과 측정

### 5. 보안 및 규정 (5%):
- 데이터 보안 (GDPR, ISO 27001)
- 산업 안전 규정
- 로봇 안전 표준

{base_examples}

**출력 형식**: 간결한 토픽 제목 리스트 (JSON 배열)
**중요**: 반드시 유효한 JSON 배열만 출력하세요.

예시:
[
  "AGV 동선 최적화를 위한 Dijkstra vs A* 알고리즘 비교",
  "사람-로봇 혼재 환경에서의 안전 구역 설계 (ISO 3691-4 기준)",
  ...
]

{num_topics}개의 전문 토픽 JSON:"""
        
        return prompt
    
    def create_topic_generation_prompt_practical(
        self,
        num_topics: int = 200,
        base_topics: List[str] = None
    ) -> str:
        """토픽 생성 프롬프트 - Practical (실무자 관점, Instruction Tuning용)"""
        
        base_examples = ""
        if base_topics:
            base_examples = f"\n기존 토픽 예시:\n" + "\n".join([f"- {t}" for t in base_topics[:5]])
        
        prompt = f"""당신은 물류 현장 실무자들의 고민을 이해하는 전문가입니다.
실무자들이 **실제로 궁금해하고 결정해야 하는** WMS/로봇자동화 질문 토픽 {num_topics}개를 생성하세요.

## 핵심 원칙:
1. **비즈니스 언어 사용**: 기술 용어보다 의사결정자/실무자가 쓰는 말
2. **질문 형태로**: "~은 어떻게?", "~얼마나?", "~가능한가?", "~나을까?" 스타일
3. **현실적 고민**: 비용, 안전, 효과, 적용 가능성, 리스크 중심

## 카테고리별 비중 (실무 중심 - 균형잡힌 분포):

### 1. 일반 WMS 운영 (30%) - 가장 많이:
- "재고 실사 시간을 어떻게 줄이나요?"
- "입출고 처리 속도 향상 방법은?"
- "재고 정확도를 높이려면?"
- "직원 교육은 얼마나 걸리나요?"
- "피크 시즌 대비 어떻게 준비하나요?"
- "반품 처리 프로세스 개선 방법"
- "창고 공간을 효율적으로 쓰려면?"
- "실시간 재고 추적이 가능한가요?"
- "오류 발생 시 대응 방법은?"
- "야간 작업 효율 관리는?"

### 2. 비용/경영/ROI (25%):
- "WMS 도입 비용은 얼마나 드나요?"
- "월 운영비용(유지보수)은?"
- "투자 회수 기간은 얼마나 걸리나요?"
- "인건비 절감 효과는 얼마나?"
- "클라우드 vs 설치형, 비용 차이는?"
- "직원 교육 비용은 얼마나?"
- "기존 시스템 교체 비용은?"
- "소규모 창고도 도입 가능한가요?"

### 3. AGV/로봇 도입 결정 (15%):
- "AGV 도입 비용과 회수 기간은?"
- "로봇 vs 사람, 실제 효율 차이는?"
- "우리 창고에 AGV 적용 가능한가?"
- "AGV 구매 vs 렌탈, 뭐가 나아?"

### 4. AGV/로봇 운영 (10%):
- "AGV와 사람이 함께 일해도 안전한가?"
- "AGV 고장나면 어떻게 대응?"
- "배터리 수명과 충전은?"

### 5. 시스템 통합/연동 (10%):
- "기존 ERP와 연동 가능한가요?"
- "엑셀 데이터 옮기기 가능해?"
- "여러 창고 통합 관리 되나요?"

### 6. 직원/조직 (10%):
- "IT 경험 없는 직원도 쓸 수 있나요?"
- "기존 직원 재교육이 필요한가요?"
- "현장 관리자 역할은 어떻게 바뀌나요?"
- "직원 저항은 어떻게 극복하나요?"

{base_examples}

**출력 형식**: 간결한 질문 형태 토픽 (JSON 배열)

**필수 준수사항**:
1. **반드시 하나의 JSON 배열만** 출력 (여러 배열 금지!)
2. 카테고리별로 나누지 말고 **하나의 배열에 섞어서** 생성
3. 전문 용어보다 일상 언어
4. "~방법론", "~알고리즘" 같은 학술 용어 지양
5. 주석이나 설명 없이 JSON만

**올바른 예시** (하나의 배열):
[
  "재고 실사 시간을 어떻게 줄이나요?",
  "WMS 도입 비용은 얼마나 드나요?",
  "AGV 도입이 우리 창고에 적합한가요?",
  "직원 교육은 얼마나 걸리나요?",
  "기존 ERP와 연동이 가능한가요?",
  ...
]

**잘못된 예시** (여러 배열 - 금지!):
[...], [...] ← 이렇게 하지 마세요!

{num_topics}개의 실무 질문 토픽 (하나의 JSON 배열로):"""
        
        return prompt
    
    def generate_personas(
        self,
        num_personas: int = 100,
        base_personas: List[Dict] = None,
        output_file: str = None
    ) -> List[Dict]:
        """페르소나 자동 생성"""
        print(f"\n{'#'*80}")
        print(f"Personas 생성 시작: {num_personas}개")
        print(f"{'#'*80}\n")
        
        # 프롬프트 생성
        prompt = self.create_persona_generation_prompt(num_personas, base_personas)
        
        # Tokenizer로 chat template 적용
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # vLLM 생성 (낮은 temperature로 안정성 확보)
        print("생성 중...", flush=True)
        
        # Persona용 매우 낮은 temperature
        sampling_params_persona = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=2500,
            repetition_penalty=1.0,
            skip_special_tokens=True
        )
        
        outputs = self.llm.generate([formatted_prompt], sampling_params_persona)
        generated_text = outputs[0].outputs[0].text
        
        # JSON 파싱
        try:
            # JSON 추출 (```json 코드 블록 처리)
            if "```json" in generated_text:
                json_start = generated_text.find("```json") + 7
                json_end = generated_text.find("```", json_start)
                generated_text = generated_text[json_start:json_end].strip()
            elif "```" in generated_text:
                json_start = generated_text.find("```") + 3
                json_end = generated_text.find("```", json_start)
                generated_text = generated_text[json_start:json_end].strip()
            
            personas = json.loads(generated_text)
            
            print(f"✓ {len(personas)}개 페르소나 생성 완료\n")
            
            # 샘플 출력
            print("생성된 페르소나 샘플 (처음 3개):")
            for i, p in enumerate(personas[:3], 1):
                print(f"\n{i}. {p.get('name', 'N/A')}")
                print(f"   배경: {p.get('background', 'N/A')[:80]}...")
                print(f"   관심사: {', '.join(p.get('concerns', [])[:3])}")
            
            # 파일 저장
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(personas, f, ensure_ascii=False, indent=2)
                
                print(f"\n✓ 저장 완료: {output_file}")
            
            return personas
        
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"생성된 텍스트:\n{generated_text[:500]}...")
            return []
    
    def generate_topics(
        self,
        num_topics: int = 200,
        base_topics: List[str] = None,
        output_file: str = None,
        topic_type: str = "mixed"  # "technical", "practical", "mixed"
    ) -> List[str]:
        """토픽 자동 생성 (technical: 답변품질용, practical: 질문자연스러움용)"""
        
        type_desc = {
            "technical": "Technical (답변 품질 향상용)",
            "practical": "Practical (실무자 질문 스타일)",
            "mixed": "Mixed (Technical + Practical)"
        }
        
        print(f"\n{'#'*80}")
        print(f"Topics 생성 시작: {num_topics}개 - {type_desc.get(topic_type, topic_type)}")
        print(f"{'#'*80}\n")
        
        # 프롬프트 생성
        if topic_type == "technical":
            prompt = self.create_topic_generation_prompt_technical(num_topics, base_topics)
        elif topic_type == "practical":
            prompt = self.create_topic_generation_prompt_practical(num_topics, base_topics)
        elif topic_type == "mixed":
            # Mixed: 절반씩 생성
            print(f"  → Technical: {num_topics//2}개")
            print(f"  → Practical: {num_topics - num_topics//2}개\n")
            
            # Technical 먼저
            topics_tech = self._generate_topics_internal(
                num_topics // 2,
                base_topics,
                self.create_topic_generation_prompt_technical(num_topics // 2, base_topics)
            )
            
            # Practical 다음
            topics_prac = self._generate_topics_internal(
                num_topics - num_topics // 2,
                base_topics,
                self.create_topic_generation_prompt_practical(num_topics - num_topics // 2, base_topics)
            )
            
            topics = topics_tech + topics_prac
            
            # 파일 저장 (mixed는 별도 처리)
            if output_file:
                self._save_topics(topics, output_file, topic_type="mixed")
            
            return topics
        else:
            raise ValueError(f"Unknown topic_type: {topic_type}")
        
        # Single type 생성
        topics = self._generate_topics_internal(num_topics, base_topics, prompt)
        
        # 파일 저장
        if output_file:
            self._save_topics(topics, output_file, topic_type=topic_type)
        
        return topics
    
    def _generate_topics_internal(
        self,
        num_topics: int,
        base_topics: List[str],
        prompt: str
    ) -> List[str]:
        """토픽 생성 내부 로직"""
        # Tokenizer로 chat template 적용
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # vLLM 생성 (낮은 temperature로 안정성 확보)
        print("생성 중...", flush=True)
        
        # Topic용 매우 낮은 temperature로 안정적인 JSON 생성
        sampling_params_topic = SamplingParams(
            temperature=0.1,  # 매우 낮게 설정
            top_p=0.9,
            max_tokens=2000,  # 약간 줄임
            repetition_penalty=1.0,  # 반복 페널티 제거
            skip_special_tokens=True
        )
        
        outputs = self.llm.generate([formatted_prompt], sampling_params_topic)
        generated_text = outputs[0].outputs[0].text
        
        # JSON 파싱
        try:
            # JSON 추출
            if "```json" in generated_text:
                json_start = generated_text.find("```json") + 7
                json_end = generated_text.find("```", json_start)
                generated_text = generated_text[json_start:json_end].strip()
            elif "```" in generated_text:
                json_start = generated_text.find("```") + 3
                json_end = generated_text.find("```", json_start)
                generated_text = generated_text[json_start:json_end].strip()
            
            topics = json.loads(generated_text)
            print(f"✓ {len(topics)}개 토픽 생성 완료\n")
            
            return topics
        
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"생성된 텍스트:\n{generated_text[:500]}...")
            return []
    
    def _save_topics(
        self,
        topics: List[str],
        output_file: str,
        topic_type: str = "mixed"
    ):
        """토픽을 파일로 저장 (카테고리 분류 포함)"""
        # 카테고리별 분류 (키워드 기반)
        categories = {
            "AGV/로봇": [],
            "운영 최적화": [],
            "시스템 통합": [],
            "비용/ROI": [],
            "기술 동향": [],
            "기타": []
        }
        
        for topic in topics:
            if any(kw in topic for kw in ["AGV", "로봇", "AMR", "코봇", "자동화"]):
                categories["AGV/로봇"].append(topic)
            elif any(kw in topic for kw in ["피킹", "재고", "크로스도킹", "풀필먼트"]):
                categories["운영 최적화"].append(topic)
            elif any(kw in topic for kw in ["ERP", "API", "통합", "클라우드", "MES"]):
                categories["시스템 통합"].append(topic)
            elif any(kw in topic for kw in ["ROI", "비용", "TCO", "CAPEX", "RaaS"]):
                categories["비용/ROI"].append(topic)
            elif any(kw in topic for kw in ["기술", "AI", "ML", "IoT", "Digital Twin", "미래"]):
                categories["기술 동향"].append(topic)
            else:
                categories["기타"].append(topic)
        
        # 카테고리별 통계
        print("카테고리별 분포:")
        for cat, items in categories.items():
            if items:
                print(f"  {cat}: {len(items)}개")
                for item in items[:2]:
                    print(f"    - {item[:80]}...")
        
        # 파일 저장
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 토픽 + 카테고리 + 타입 정보 저장
        output_data = {
            "topics": topics,
            "categories": categories,
            "total": len(topics),
            "topic_type": topic_type,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 저장 완료: {output_file}")
    
    def generate_all(
        self,
        num_personas: int = 100,
        num_topics: int = 200,
        output_dir: str = "expanded_data"
    ):
        """페르소나 + 토픽 일괄 생성"""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from personas import PERSONAS, WMS_TOPICS
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Personas 생성
        personas = self.generate_personas(
            num_personas=num_personas,
            base_personas=PERSONAS,
            output_file=output_path / f"personas_{num_personas}_{timestamp}.json"
        )
        
        # 2. Topics 생성
        topics = self.generate_topics(
            num_topics=num_topics,
            base_topics=WMS_TOPICS,
            output_file=output_path / f"topics_{num_topics}_{timestamp}.json"
        )
        
        print(f"\n{'='*80}")
        print(f"생성 완료!")
        print(f"  Personas: {len(personas)}개")
        print(f"  Topics: {len(topics)}개")
        print(f"  예상 QA 생성 가능: {len(personas) * len(topics) * 5:,}개 (×5 questions/topic)")
        print(f"{'='*80}\n")
        
        return personas, topics


# CLI 실행
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Persona & Topic Generator')
    parser.add_argument('--personas', type=int, default=100, help='생성할 페르소나 수')
    parser.add_argument('--topics', type=int, default=200, help='생성할 토픽 수')
    parser.add_argument('--output-dir', default='expanded_data', help='출력 디렉토리')
    parser.add_argument('--mode', choices=['all', 'personas', 'topics'], default='all')
    parser.add_argument(
        '--topic-type', 
        choices=['technical', 'practical', 'mixed'], 
        default='mixed',
        help='토픽 타입: technical(답변품질), practical(질문스타일), mixed(혼합)'
    )
    
    args = parser.parse_args()
    
    generator = PersonaTopicGenerator()
    
    if args.mode == 'all':
        generator.generate_all(
            num_personas=args.personas,
            num_topics=args.topics,
            output_dir=args.output_dir
        )
    elif args.mode == 'personas':
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "legacy" / "question"))
        from personas import PERSONAS
        generator.generate_personas(
            num_personas=args.personas,
            base_personas=PERSONAS,
            output_file=f"{args.output_dir}/personas_{args.personas}.json"
        )
    elif args.mode == 'topics':
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "legacy" / "question"))
        from personas import WMS_TOPICS
        generator.generate_topics(
            num_topics=args.topics,
            base_topics=WMS_TOPICS,
            output_file=f"{args.output_dir}/topics_{args.topics}_{args.topic_type}.json",
            topic_type=args.topic_type
        )


"""
WMS Q-A Demo - 최적화 버전
개선사항:
1. 답변 반복 문제 해결 (repetition_penalty=1.2)
2. 토큰 제한 증가 (512 → 768)
3. 개선된 질문 파싱 (정규식)
4. FAISS GPU 전환 시도
5. 답변 후처리 (중복 제거)
"""

import torch
import json
import faiss
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


def load_faiss_index_with_gpu():
    """FAISS 인덱스 로딩 및 GPU 전환 시도"""
    print(f"Loading FAISS index...", flush=True)
    
    # FAISS 인덱스 로드
    index_path = "/home/work/tesseract/faiss_storage/warehouse_automation_knowledge.index"
    index = faiss.read_index(index_path)
    print(f"✓ FAISS index loaded: {index.ntotal} vectors", flush=True)
    
    # GPU 전환 시도
    if torch.cuda.is_available():
        try:
            print(f"Attempting to move FAISS index to GPU...", flush=True)
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"✓ FAISS moved to GPU! (10-50x faster)", flush=True)
            index = gpu_index
        except Exception as e:
            print(f"⚠ GPU transfer failed: {e}", flush=True)
            print(f"✓ Using CPU FAISS (still fast)", flush=True)
    else:
        print(f"✓ Using CPU FAISS", flush=True)
    
    # 문서 로드
    with open("/home/work/tesseract/faiss_storage/documents.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"✓ Documents loaded: {len(documents)} docs\n", flush=True)
    
    # 임베딩 모델 로드
    print(f"Loading embedding model...", flush=True)
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask", device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Embedding model loaded\n", flush=True)
    
    return index, documents, embed_model


def search_faiss(query, index, documents, embed_model, k=3):
    """FAISS로 검색"""
    # 쿼리 임베딩
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype('float32')
    
    # 검색
    distances, indices = index.search(query_vec, k)
    
    # 결과 수집
    results = []
    for idx in indices[0]:
        if idx < len(documents):
            results.append(documents[idx])
    
    return results


def parse_questions_improved(response):
    """개선된 질문 파싱 (정규식 사용)"""
    # Q1, Q2, Q1:, Q1), Q1. 등 모든 패턴 처리
    pattern = r'Q\d+[\):\.]\s*(.+?)(?=Q\d+|$)'
    matches = re.findall(pattern, response, re.DOTALL)
    
    questions = []
    for match in matches:
        question = match.strip()
        # 줄바꿈 제거하고 공백 정리
        question = ' '.join(question.split())
        if len(question) > 10:  # 최소 길이
            questions.append(question)
    
    return questions


def remove_repetitions(text):
    """답변에서 반복 문장 제거"""
    sentences = text.split('.')
    unique = []
    seen = set()
    
    for sent in sentences:
        # 정규화 (소문자, 공백 제거)
        normalized = sent.strip().lower()
        normalized = ' '.join(normalized.split())
        
        # 충분히 긴 문장이고 아직 보지 못한 경우만 추가
        if normalized not in seen and len(normalized) > 20:
            unique.append(sent.strip())
            seen.add(normalized)
    
    result = '. '.join(unique)
    if result and not result.endswith('.'):
        result += '.'
    
    return result


def generate_questions_with_exaone(topic="WMS 도입"):
    """EXAONE으로 WMS 도입 관련 질문 생성 (개선)"""
    print(f"\n{'='*80}", flush=True)
    print(f"STEP 1: Question Generation with EXAONE (Optimized)", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # FAISS 로드
    index, documents, embed_model = load_faiss_index_with_gpu()
    
    print(f"Searching for context about: {topic}", flush=True)
    search_results = search_faiss(topic, index, documents, embed_model, k=3)
    context = "\n\n".join([doc[:500] for doc in search_results])
    print(f"✓ Retrieved {len(search_results)} contexts\n", flush=True)
    
    # EXAONE 모델 로딩
    print(f"Loading EXAONE model...", flush=True)
    model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print(f"✓ EXAONE loaded ({model.get_memory_footprint() / 1024**3:.2f} GB)\n", flush=True)
    
    # 질문 생성 프롬프트
    prompt = f"""당신은 물류 시스템 전문가입니다. WMS(Warehouse Management System) 도입에 관해 다양한 질문을 생성하세요.

참고 컨텍스트:
{context}

WMS 도입과 관련하여 5개의 전문적인 질문을 생성하세요.

다음 형식으로 출력하세요:
Q1: [질문]
Q2: [질문]
Q3: [질문]
Q4: [질문]
Q5: [질문]"""
    
    # 메시지 형식으로 변환
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 질문 생성
    print(f"Generating questions...", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"\n{'─'*80}", flush=True)
    print(f"Generated Questions:", flush=True)
    print(f"{'─'*80}", flush=True)
    print(response, flush=True)
    print(f"{'─'*80}\n", flush=True)
    
    # 개선된 질문 파싱
    questions = parse_questions_improved(response)
    
    print(f"✓ Parsed {len(questions)} questions", flush=True)
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q[:80]}...", flush=True)
    print()
    
    # 메모리 정리
    del model
    del tokenizer
    del embed_model
    torch.cuda.empty_cache()
    
    return questions, index, documents


def generate_answers_with_eeve(questions, index, documents):
    """EEVE로 RAG 기반 답변 생성 (개선)"""
    print(f"\n{'='*80}", flush=True)
    print(f"STEP 2: Answer Generation with EEVE (Optimized)", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # 임베딩 모델 로드
    print(f"Loading embedding model...", flush=True)
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask", device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Embedding model loaded\n", flush=True)
    
    # EEVE 모델 로딩
    print(f"Loading EEVE model...", flush=True)
    model_name = "MyeongHo0621/eeve-vss-smh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print(f"✓ EEVE loaded ({model.get_memory_footprint() / 1024**3:.2f} GB)\n", flush=True)
    
    qa_pairs = []
    
    # 각 질문에 대해 답변 생성
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*80}", flush=True)
        print(f"Question {i}/{len(questions)}: {question}", flush=True)
        print(f"{'─'*80}\n", flush=True)
        
        # FAISS에서 관련 컨텍스트 검색
        print(f"Retrieving relevant contexts...", flush=True)
        search_results = search_faiss(question, index, documents, embed_model, k=5)
        
        # 컨텍스트 결합
        contexts = "\n\n".join([
            f"[참고자료 {idx+1}]\n{doc[:400]}"
            for idx, doc in enumerate(search_results)
        ])
        print(f"✓ Retrieved {len(search_results)} contexts\n", flush=True)
        
        # RAG 프롬프트 생성 (실무자 경험 기반)
        rag_prompt = f"""당신은 10년 이상 물류 시스템 구축 프로젝트를 진행해온 현장 전문가입니다.
여러 고객사의 WMS 도입 프로젝트를 성공시킨 경험이 있습니다.

아래는 실제 프로젝트 사례와 현장 경험 자료입니다:
{contexts}

질문: {question}

위 사례들을 바탕으로 실무 경험을 공유하듯이 답변해주세요.

답변 작성 규칙:
1. "참고자료에 따르면" (X) → "실제 제가 진행한 프로젝트에서는", "경험상", "보통 ~합니다" (O)
2. 구체적인 숫자와 기간을 포함 (예: "평균 3-6개월", "약 30% 절감", "하루 500건 → 1,200건")
3. 실제 고객사 사례 느낌으로 (예: "A사의 경우", "중소 물류센터들은 보통")
4. "해야 합니다", "필수적입니다" 같은 교과서 표현 금지
5. 자연스러운 대화체로, 상담하듯이 설명
6. 장단점을 솔직하게 (좋은 점만 말하지 말고 주의할 점도 언급)

나쁜 답변:
"참고자료에 따르면 WMS 시스템은 재고 관리 효율성을 향상시킵니다."

좋은 답변:
"제가 작년에 진행한 중소 물류센터 프로젝트를 보면요, 재고 실사 시간이 하루 8시간에서 2시간으로 줄었습니다. 
다만 처음 3개월은 직원들이 적응하느라 오히려 더 느렸어요."

답변:"""
        
        # EEVE 프롬프트 템플릿
        full_prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {rag_prompt}
Assistant: """
        
        # 토크나이징
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        # 답변 생성 (개선된 파라미터)
        import time
        print(f"Generating answer...", flush=True)
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,  # 512 → 768
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.15,  # 1.0 → 1.15 (최적 균형)
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        answer = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 답변 후처리 (반복 제거)
        answer_cleaned = remove_repetitions(answer)
        
        print(f"\nAnswer (cleaned):", flush=True)
        print(f"{answer_cleaned}\n", flush=True)
        print(f"Inference time: {inference_time:.2f} sec", flush=True)
        print(f"Original length: {len(answer)} chars", flush=True)
        print(f"Cleaned length: {len(answer_cleaned)} chars", flush=True)
        print(f"Reduction: {len(answer) - len(answer_cleaned)} chars\n", flush=True)
        
        qa_pairs.append({
            'question': question,
            'answer': answer_cleaned,
            'answer_original': answer,
            'num_contexts': len(search_results),
            'inference_time': inference_time,
            'tokens_generated': outputs.shape[1] - inputs['input_ids'].shape[1]
        })
    
    # 메모리 정리
    del model
    del tokenizer
    del embed_model
    torch.cuda.empty_cache()
    
    return qa_pairs


def main():
    """WMS Q-A 데모 실행 (최적화)"""
    print(f"\n{'#'*80}", flush=True)
    print(f"WMS Q-A Generation Demo - OPTIMIZED VERSION", flush=True)
    print(f"{'#'*80}\n", flush=True)
    
    # Step 1: EXAONE으로 질문 생성
    questions, index, documents = generate_questions_with_exaone("WMS 시스템 도입 및 구현")
    
    if not questions:
        print("❌ No questions generated. Exiting...", flush=True)
        return
    
    print(f"\n✓ Generated {len(questions)} questions", flush=True)
    
    # Step 2: EEVE로 답변 생성 (처음 3개만)
    questions_to_answer = questions[:3]
    print(f"\nAnswering first {len(questions_to_answer)} questions...\n", flush=True)
    
    qa_pairs = generate_answers_with_eeve(questions_to_answer, index, documents)
    
    # 결과 출력
    print(f"\n{'#'*80}", flush=True)
    print(f"Final Results (Optimized)", flush=True)
    print(f"{'#'*80}\n", flush=True)
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{'='*80}", flush=True)
        print(f"Q-A Pair {i}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"\nQ: {qa['question']}", flush=True)
        print(f"\nA: {qa['answer'][:500]}...", flush=True)
        print(f"\nStats:", flush=True)
        print(f"  - RAG contexts: {qa['num_contexts']}", flush=True)
        print(f"  - Inference time: {qa['inference_time']:.2f}s", flush=True)
        print(f"  - Tokens generated: {qa['tokens_generated']}", flush=True)
        print(f"  - Chars saved: {len(qa['answer_original']) - len(qa['answer'])}", flush=True)
    
    # JSON으로 저장
    output_file = "/home/work/tesseract/datageneration/test/wms_qa_optimized_result.json"
    
    # 저장용 데이터 (original 제외)
    save_data = [{
        'question': qa['question'],
        'answer': qa['answer'],
        'num_contexts': qa['num_contexts'],
        'inference_time': qa['inference_time'],
        'tokens_generated': qa['tokens_generated']
    } for qa in qa_pairs]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'#'*80}", flush=True)
    print(f"✓ Results saved to: {output_file}", flush=True)
    print(f"✓ Total Q-A pairs: {len(qa_pairs)}", flush=True)
    
    # 통계 요약
    avg_inference = sum(qa['inference_time'] for qa in qa_pairs) / len(qa_pairs)
    avg_tokens = sum(qa['tokens_generated'] for qa in qa_pairs) / len(qa_pairs)
    total_saved = sum(len(qa['answer_original']) - len(qa['answer']) for qa in qa_pairs)
    
    print(f"\nPerformance Summary:", flush=True)
    print(f"  - Average inference time: {avg_inference:.2f}s", flush=True)
    print(f"  - Average tokens generated: {avg_tokens:.0f}", flush=True)
    print(f"  - Total repetitions removed: {total_saved} chars", flush=True)
    print(f"{'#'*80}\n", flush=True)


if __name__ == "__main__":
    main()


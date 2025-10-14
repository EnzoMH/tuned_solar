#!/usr/bin/env python3
"""
토크나이저 → 임베딩 매핑 테스트
학습 전후 비교 가능
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
# from unsloth import FastLanguageModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class TokenEmbeddingTester:
    """토큰 임베딩 매핑 테스트"""
    
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Args:
            model_path: 모델 경로 (None이면 초기 상태 테스트)
            tokenizer_path: 토크나이저 경로
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """모델 & 토크나이저 로드"""
        print("="*80)
        print("모델 로딩")
        print("="*80)
        
        if self.model_path:
            # 학습된 모델 로드
            print(f"학습된 모델: {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        else:
            # ⭐ Unsloth 대신 Transformers 직접 사용
            print("초기 모델: Qwen (BF16 버전)")
            
            # ⚠️ FP8 버전 대신 원본 BF16 버전 사용
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-32B-Instruct",  # FP8 아닌 원본
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # 한국어 최적화 토크나이저
            tokenizer_name = self.tokenizer_path or os.getenv("TOKENIZER")
            if not tokenizer_name:
                raise ValueError("TOKENIZER 환경변수가 설정되지 않았습니다.")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            
            # 임베딩 크기 조정
            original_size = self.model.get_input_embeddings().weight.shape[0]
            new_size = len(self.tokenizer)
            
            if original_size != new_size:
                print(f"임베딩 크기 조정: {original_size:,} → {new_size:,}")
                self.model.resize_token_embeddings(new_size)
        
        print("✓ 로드 완료")
        print(f"  Vocab size: {len(self.tokenizer):,}")
        print(f"  Embedding shape: {self.model.get_input_embeddings().weight.shape}")
        print("="*80)
    
    def test_token_to_embedding(self, text_samples):
        """토큰 → 임베딩 매핑 테스트"""
        print("\n" + "="*80)
        print("🔍 토큰 → 임베딩 매핑 테스트")
        print("="*80)
        
        embed_layer = self.model.get_input_embeddings()
        
        for text in text_samples:
            print(f"\n📝 텍스트: '{text}'")
            print("-"*80)
            
            # 토크나이징
            tokens = self.tokenizer(text, return_tensors="pt")
            token_ids = tokens["input_ids"][0]
            
            print(f"토큰 개수: {len(token_ids)}")
            
            # 각 토큰 정보
            for idx, token_id in enumerate(token_ids):
                token_id = token_id.item()
                token_str = self.tokenizer.decode([token_id])
                
                # 임베딩 벡터 (BFloat16 → Float32 → NumPy)
                embedding = embed_layer.weight[token_id].detach().cpu().float().numpy()
                
                print(f"\n  [{idx}] Token ID: {token_id}")
                print(f"      Token: '{token_str}'")
                print(f"      Embedding 통계:")
                print(f"        평균: {embedding.mean():.6f}")
                print(f"        표준편차: {embedding.std():.6f}")
                print(f"        최소: {embedding.min():.6f}")
                print(f"        최대: {embedding.max():.6f}")
    
    def compare_token_embeddings(self, token1, token2):
        """두 토큰의 임베딩 유사도 비교"""
        print("\n" + "="*80)
        print(f"🔬 토큰 임베딩 유사도: '{token1}' vs '{token2}'")
        print("="*80)
        
        embed_layer = self.model.get_input_embeddings()
        
        # 토큰 ID
        token1_id = self.tokenizer.encode(token1, add_special_tokens=False)[0]
        token2_id = self.tokenizer.encode(token2, add_special_tokens=False)[0]
        
        # 임베딩 (BFloat16 → Float32 → NumPy)
        emb1 = embed_layer.weight[token1_id].detach().cpu().float().numpy()
        emb2 = embed_layer.weight[token2_id].detach().cpu().float().numpy()
        
        # 유사도
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        distance = np.linalg.norm(emb1 - emb2)
        
        print(f"Token 1: '{token1}' (ID: {token1_id})")
        print(f"  평균: {emb1.mean():.6f}, 표준편차: {emb1.std():.6f}")
        print(f"\nToken 2: '{token2}' (ID: {token2_id})")
        print(f"  평균: {emb2.mean():.6f}, 표준편차: {emb2.std():.6f}")
        print(f"\n코사인 유사도: {similarity:.6f}")
        print(f"유클리드 거리: {distance:.6f}")
        
        if similarity > 0.9:
            print("→ 매우 유사함 ✅")
        elif similarity > 0.7:
            print("→ 유사함")
        elif similarity > 0.3:
            print("→ 약간 유사함")
        else:
            print("→ 거의 다름 ❌")
    
def main():
    """메인 함수"""
    
    print("\n" + "🔬"*40)
    print("토큰 → 임베딩 매핑 테스트 (학습 전)")
    print("🔬"*40)
    
    # .env 파일에서 TOKENIZER 환경변수 사용
    tester = TokenEmbeddingTester(
        model_path=None,
        tokenizer_path=None  # .env의 TOKENIZER 사용
    )
    
    tester.load_model()
    
    # 샘플 테스트
    test_samples = [
        "안녕하세요",
        "데이터 분석",
        "인공지능",
    ]
    tester.test_token_to_embedding(test_samples)
    
    # 유사도 비교
    tester.compare_token_embeddings("데이터", "분석")
    tester.compare_token_embeddings("안녕", "감사")
    
    print("\n" + "="*80)
    print("✅ 테스트 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
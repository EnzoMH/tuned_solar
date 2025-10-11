"""FAISS 벡터 DB 구축"""

import torch
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import config


class VectorDBBuilder:
    """크롤링 데이터로 FAISS 벡터 DB 구축"""
    
    def __init__(self):
        print("\n임베딩 모델 로딩...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✓ 임베딩 모델 로딩 완료: {config.EMBEDDING_MODEL}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """크롤링 데이터 로딩"""
        print(f"\n문서 로딩 중: {config.CRAWLED_DIR}")
        
        documents = []
        crawled_path = Path(config.CRAWLED_DIR)
        
        # .txt 파일들
        txt_files = list(crawled_path.glob("*.txt"))
        for file_path in txt_files:
            print(f"  읽는 중: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": file_path.name}
                        ))
            except Exception as e:
                print(f"  ❌ 오류: {e}")
        
        # TODO: .csv, .json 파일 처리도 추가 가능
        
        print(f"\n총 {len(documents)}개 문서 로딩 완료")
        return documents
    
    def build_vectordb(self) -> FAISS:
        """벡터 DB 구축"""
        print("\n" + "=" * 80)
        print(" FAISS 벡터 DB 구축")
        print("=" * 80)
        
        # 문서 로딩
        documents = self.load_documents()
        
        if not documents:
            raise ValueError(
                f"문서가 없습니다!\n"
                f"{config.CRAWLED_DIR}에 .txt 파일들을 추가하세요."
            )
        
        # 청크 분할
        print(f"\n문서 청크 분할 중 (chunk_size={config.CHUNK_SIZE})...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ 총 {len(chunks)}개 청크 생성")
        
        # FAISS 구축
        print("\nFAISS 벡터 DB 구축 중...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("✓ 벡터 DB 구축 완료")
        
        # 저장
        print(f"\n벡터 DB 저장 중: {config.FAISS_PATH}")
        vectorstore.save_local(config.FAISS_PATH)
        print("✓ 저장 완료")
        
        return vectorstore
    
    @staticmethod
    def load_vectordb() -> FAISS:
        """기존 벡터 DB 로딩"""
        print(f"\nFAISS 벡터 DB 로딩: {config.FAISS_PATH}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = FAISS.load_local(
            config.FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("✓ FAISS 로드 완료")
        return vectorstore


if __name__ == "__main__":
    builder = VectorDBBuilder()
    builder.build_vectordb()



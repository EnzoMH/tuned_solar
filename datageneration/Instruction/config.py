import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """프로젝트 설정"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # SOLAR 모델
    SOLAR_BASE_MODEL = "upstage/SOLAR-10.7B-v1.0"
    SOLAR_ADAPTER = "MyeongHo0621/solar-korean-wms"
    
    # 경로
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CRAWLED_DIR = DATA_DIR / "crawled"
    OUTPUT_DIR = DATA_DIR / "output"
    
    # FAISS 설정 (프로젝트 루트의 faiss_storage 사용)
    PROJECT_ROOT = BASE_DIR.parent
    FAISS_DIR = PROJECT_ROOT / "faiss_storage"
    FAISS_INDEX_FILE = "warehouse_automation_knowledge.index"
    FAISS_CONFIG_FILE = "config.json"
    FAISS_DOCUMENTS_FILE = "documents.json"
    FAISS_METADATA_FILE = "metadata.json"
    EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # 생성 설정
    QUESTIONS_PER_TOPIC = 2
    MAX_ANSWER_TOKENS = 600  # 더 긴 답변을 위해 증가
    
    # SOLAR 생성 파라미터
    SOLAR_TEMPERATURE = 1.0  # 일관성 있는 답변 (0.3 -> 1.0)
    SOLAR_TOP_P = 0.85
    SOLAR_TOP_K = 30
    SOLAR_REPETITION_PENALTY = 1.15
    
    @classmethod
    def validate(cls):
        """설정 검증"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다!\n"
                "1. https://makersuite.google.com/app/apikey 에서 키 발급\n"
                "2. .env 파일에 GEMINI_API_KEY=your_key 추가"
            )
        
        # 디렉토리 생성
        cls.CRAWLED_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        return True

config = Config()


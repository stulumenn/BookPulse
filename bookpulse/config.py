from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_XLSX = DATA_DIR / "B2B_Transaction_Data_.xlsx"
CLEAN_FILE = DATA_DIR / "transactions_clean.csv.gz"


SALES_DOC_TYPES = ["BAD", "BSR", "DPE", "DPF"]

MIN_UNIT_PRICE = 0.01

MERGE_CUSTOMER_BRANCHES = True

LIMITED_MIN_ACTIVE_MONTHS = 3   
LIMITED_MIN_TOTAL_QTY = 10      
VOLUME_THRESHOLD = None
CV_THRESHOLD = None


NEXT_CALLS_WEIGHTS = {
    "category_qty": 0.15,      
    "category_share": 0.10,     
    "frequency": 0.20,         
    "diversity": 0.10,         
    "similar_items": 0.35,      
    "recency": 0.10,            
}
DORMANT_DAYS = 60              

OLLAMA_LOCAL_HOST = "http://localhost:11434"
OLLAMA_CLOUD_HOST = "https://ollama.com"
OLLAMA_LOCAL_MODEL = "qwen2.5:7b"   
OLLAMA_CLOUD_MODEL = "gemma4:31b"  
OPENAI_MODEL_DEFAULT = "gpt-6-luna"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-5-5"
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash"

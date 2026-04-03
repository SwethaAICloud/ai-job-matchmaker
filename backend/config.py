import os
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "data.csv"
FAISS_INDEX_PATH = "faiss_index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

JOB_LABELS = {
    0: 'Software_Developer',
    1: 'Front_End_Developer',
    2: 'Network_Administrator',
    3: 'Web_Developer',
    4: 'Project_manager',
    5: 'Database_Administrator',
    6: 'Security_Analyst',
    7: 'Systems_Administrator',
    8: 'Python_Developer',
    9: 'Java_Developer'
}

JOB_COLUMNS = list(JOB_LABELS.values())

TEXT_COLUMNS = [
    'Text', 'Skills', 'Education',
    'Experience', 'Additional_Information'
]

GROQ_MODELS = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'gemma2-9b-it',
    'mixtral-8x7b-32768',
]

print("Config loaded")
if GROQ_API_KEY:
    print("   Groq key: Found")
else:
    print("   WARNING: No GROQ_API_KEY in .env")
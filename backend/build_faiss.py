import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, DATA_PATH, JOB_COLUMNS

print("=" * 50)
print("   BUILDING IMPROVED FAISS INDEX")
print("=" * 50)

# Step 1: Load data
print("\n1. Loading data...")
df = pd.read_csv(DATA_PATH, nrows=5000)
print("   Rows: " + str(len(df)))

for col in ['Text', 'Skills', 'Education', 'Experience', 'Additional_Information']:
    if col in df.columns:
        df[col] = df[col].fillna('')

# Step 2: Create documents
print("\n2. Creating documents...")
docs = []

for idx, row in df.iterrows():
    jobs = [j for j in JOB_COLUMNS if row.get(j, 0) == 1]
    skills = str(row.get('Skills', ''))
    education = str(row.get('Education', ''))
    experience = str(row.get('Experience', ''))
    additional = str(row.get('Additional_Information', ''))

    full_text = (
        str(row.get('Text', '')) + " " +
        skills + " " + education + " " +
        experience + " " + additional
    ).lower().strip()

    # Full resume document
    docs.append(Document(
        page_content=full_text[:1500],
        metadata={
            'doc_id': str(idx),
            'jobs': ', '.join(jobs),
            'skills': skills[:300],
            'education': education[:200],
            'type': 'resume_full'
        }
    ))

    # Skills-only document
    if skills.strip():
        skill_text = "skills: " + skills.lower() + " job role: " + ' '.join(jobs).lower().replace('_', ' ')
        docs.append(Document(
            page_content=skill_text[:500],
            metadata={
                'doc_id': str(idx) + '_skills',
                'jobs': ', '.join(jobs),
                'skills': skills[:300],
                'type': 'resume_skills'
            }
        ))

    # Experience-only document
    if experience.strip() and len(experience) > 20:
        exp_text = "experience: " + experience.lower() + " job role: " + ' '.join(jobs).lower().replace('_', ' ')
        docs.append(Document(
            page_content=exp_text[:500],
            metadata={
                'doc_id': str(idx) + '_exp',
                'jobs': ', '.join(jobs),
                'type': 'resume_experience'
            }
        ))

# Add job profile documents
print("\n3. Creating job profiles...")
for job_name in JOB_COLUMNS:
    if job_name not in df.columns:
        continue
    mask = df[job_name] == 1
    job_df = df[mask]
    if len(job_df) == 0:
        continue

    all_skills = " ".join(job_df['Skills'].fillna('').tolist())[:2000]

    profile_texts = [
        "job role " + job_name.replace('_', ' ') + " requires these skills: " + all_skills[:500],
        "people working as " + job_name.replace('_', ' ') + " typically know: " + all_skills[:500],
        job_name.replace('_', ' ') + " developer programmer engineer skills: " + all_skills[:300],
    ]

    for j, text in enumerate(profile_texts):
        docs.append(Document(
            page_content=text.lower(),
            metadata={
                'doc_id': 'profile_' + job_name + '_' + str(j),
                'jobs': job_name,
                'type': 'job_profile'
            }
        ))

    print("   " + job_name + ": " + str(int(mask.sum())) + " resumes")

print("\n   Total documents: " + str(len(docs)))

# Step 3: Chunk
print("\n4. Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", ", ", " "]
)
chunks = splitter.split_documents(docs)
print("   Chunks: " + str(len(chunks)))

# Step 4: Embed
print("\n5. Embedding (10-15 minutes)...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

store = FAISS.from_documents(documents=chunks, embedding=embeddings)

# Step 5: Save
print("\n6. Saving...")
store.save_local("faiss_index")

if os.path.exists("faiss_index/index.faiss"):
    size = os.path.getsize("faiss_index/index.faiss")
    print("   Size: " + str(round(size / 1024 / 1024, 1)) + " MB")
    print("\n" + "=" * 50)
    print("   SUCCESS!")
    print("   Now run: python app.py")
    print("=" * 50)
else:
    print("   FAILED - index not created")
    
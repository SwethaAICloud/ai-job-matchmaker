# ============================================================
# 7_full_pipeline.py
# Complete RAG Pipeline: Load → Chunk → Embed → Store → Search → Answer
# ============================================================

import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from groq import Groq

from config import GROQ_API_KEY, EMBEDDING_MODEL, JOB_COLUMNS
from data_loader import load_data, get_job_documents


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


# ============================================================
# STEP 1: Load Data
# ============================================================

def step1_load():
    print_header("STEP 1: LOAD DATA")
    df, documents = load_data()
    job_docs = get_job_documents(df, documents)
    print(f"\n   Resumes: {len(documents)}")
    print(f"   Job profiles: {len(job_docs)}")
    return df, documents, job_docs


# ============================================================
# STEP 2: Create LangChain Documents
# ============================================================

def step2_documents(documents, job_docs):
    print_header("STEP 2: CREATE LANGCHAIN DOCUMENTS")
    
    lc_docs = []
    for doc in documents:
        lc_docs.append(Document(
            page_content=doc['text'],
            metadata={
                'doc_id': doc['id'],
                'jobs': doc['jobs_str'],
                'skills': doc.get('skills', '')[:300],
                'type': 'resume'
            }
        ))
    
    for jd in job_docs:
        lc_docs.append(Document(
            page_content=jd['text'],
            metadata={
                'doc_id': jd['id'],
                'jobs': jd.get('job_name', ''),
                'type': 'job_profile'
            }
        ))
    
    print(f"   Total documents: {len(lc_docs)}")
    print(f"   Sample: {lc_docs[0].page_content[:100]}...")
    return lc_docs


# ============================================================
# STEP 3: Chunk Documents
# ============================================================

def step3_chunk(lc_docs):
    print_header("STEP 3: CHUNK DOCUMENTS (Recursive)")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    chunks = splitter.split_documents(lc_docs)
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    print(f"   Sample chunk: {chunks[0].page_content[:100]}...")
    return chunks


# ============================================================
# STEP 4: Create Embeddings + FAISS Store
# ============================================================

def step4_embed_store(chunks):
    print_header("STEP 4: EMBED + STORE IN FAISS")
    
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   This takes 15-30 minutes...\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    store.save_local("faiss_index")
    
    print(f"   FAISS index saved!")
    print(f"   Chunks indexed: {len(chunks)}")
    return store, embeddings


# ============================================================
# STEP 5: Create Retriever
# ============================================================

def step5_retriever(store):
    print_header("STEP 5: CREATE RETRIEVER")
    
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    print("   Retriever: MMR (Maximum Marginal Relevance)")
    print("   k=5, fetch_k=10, lambda=0.5")
    print("   This ensures DIVERSE context for LLM")
    return retriever


# ============================================================
# STEP 6: Create LLM + Chain
# ============================================================

def step6_chain(retriever):
    print_header("STEP 6: CREATE LLM + CHAIN")
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
    )
    
    prompt = PromptTemplate(
        template="""You are an AI Job Matchmaker with 29,000+ IT resumes.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}

Instructions:
- Recommend matching job roles with reasoning
- Use retrieved data as evidence
- Be specific and encouraging
- Available roles: Software Developer, Front End Developer, Network Administrator,
  Web Developer, Project Manager, Database Administrator, Security Analyst,
  Systems Administrator, Python Developer, Java Developer

ANSWER:""",
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("   LLM: Groq (llama-3.1-8b-instant)")
    print("   Chain: RetrievalQA (stuff method)")
    print("   Prompt: Custom job matching template")
    return chain


# ============================================================
# STEP 7: Test the Pipeline
# ============================================================

def step7_test(chain):
    print_header("STEP 7: TEST THE PIPELINE")
    
    test_questions = [
        "Which role suits me if I know Python and SQL?",
        "I have 3 years React experience, what job?",
        "Compare frontend vs backend for me",
    ]
    
    for q in test_questions:
        print(f"\n   Q: {q}")
        print("   " + "-" * 50)
        
        result = chain.invoke({"query": q})
        answer = result.get('result', 'No answer')
        sources = result.get('source_documents', [])
        
        print(f"   A: {answer[:200]}...")
        print(f"   Sources: {len(sources)} documents")
        
        time.sleep(4)
    
    print("\n   Pipeline working!")


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("COMPLETE RAG PIPELINE")
    print("""
   Pipeline Flow:
   
   CSV Data
     ↓
   [STEP 1] Load Data (pandas)
     ↓
   [STEP 2] Create Documents (LangChain)
     ↓
   [STEP 3] Chunk (RecursiveCharacterTextSplitter)
     ↓
   [STEP 4] Embed + Store (SentenceTransformer → FAISS)
     ↓
   [STEP 5] Create Retriever (MMR)
     ↓
   [STEP 6] Create Chain (Groq LLM + Prompt)
     ↓
   [STEP 7] Test!
    """)
    
    print("  1. Build Full Pipeline (15-30 min)")
    print("  2. Load Existing + Test")
    print("  3. Exit\n")
    
    choice = input("  Choice: ").strip()
    
    if choice == '1':
        df, documents, job_docs = step1_load()
        lc_docs = step2_documents(documents, job_docs)
        chunks = step3_chunk(lc_docs)
        store, embeddings = step4_embed_store(chunks)
        retriever = step5_retriever(store)
        chain = step6_chain(retriever)
        step7_test(chain)
        
    elif choice == '2':
        print("\n   Loading existing FAISS index...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        store = FAISS.load_local(
            "faiss_index", embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = step5_retriever(store)
        chain = step6_chain(retriever)
        step7_test(chain)


if __name__ == "__main__":
    main()
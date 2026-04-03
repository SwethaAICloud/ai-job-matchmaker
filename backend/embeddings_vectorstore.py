# ============================================================
# 3_embeddings_vectorstore.py
# Embeddings + Vector Store (FAISS)
# ============================================================

import numpy as np
import time
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


# ============================================================
# PART 1: What Are Embeddings?
# ============================================================

def demo_embeddings():
    print_header("PART 1: EMBEDDINGS — Text to Numbers")
    
    print("""
   What are embeddings?
   Text → Neural Network → Array of numbers (vector)
   
   "Python Developer" → [0.23, -0.45, 0.12, ...] (384 numbers)
   
   WHY? Computers can't understand text.
   But they CAN compare numbers.
   Similar text → Similar numbers → Easy to find matches!
    """)
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Demo texts
    texts = [
        "Python developer with Django experience",
        "Backend programmer skilled in Python and Flask",
        "React frontend developer with JavaScript",
        "Network administrator managing Cisco routers",
        "Python data scientist using Pandas and NumPy",
    ]
    
    print("   Generating embeddings...")
    embeddings = model.encode(texts)
    
    print(f"\n   Model: {EMBEDDING_MODEL}")
    print(f"   Dimension: {embeddings.shape[1]}")
    print(f"   Texts embedded: {len(texts)}")
    
    print("\n   EMBEDDINGS (first 8 values):")
    print("   " + "-" * 55)
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        values = ", ".join([f"{v:.3f}" for v in emb[:8]])
        print(f"   {i+1}. \"{text[:40]}...\"")
        print(f"      [{values}, ...]")
        print()
    
    # Show similarity
    print("   SIMILARITY MATRIX:")
    print("   " + "-" * 55)
    print("   (Higher = more similar, 1.0 = identical)\n")
    
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Print header
    print("   " + " " * 5, end="")
    for i in range(len(texts)):
        print(f"  T{i+1}   ", end="")
    print()
    
    for i in range(len(texts)):
        print(f"   T{i+1}", end=" ")
        for j in range(len(texts)):
            sim = sim_matrix[i][j]
            if sim > 0.7:
                print(f" {sim:.2f}*", end=" ")
            else:
                print(f" {sim:.2f} ", end=" ")
        short = texts[i][:30]
        print(f"  {short}")
    
    print("\n   * = High similarity (> 0.7)")
    print("\n   KEY INSIGHT:")
    print("   T1 (Python Django) and T2 (Python Flask) are SIMILAR (both backend Python)")
    print("   T1 (Python) and T3 (React) are DIFFERENT (backend vs frontend)")
    
    return model, embeddings


# ============================================================
# PART 2: Vector Store (FAISS)
# ============================================================

def demo_vectorstore(model):
    print_header("PART 2: VECTOR STORE — FAISS")
    
    print("""
   What is a Vector Store?
   A database optimized for storing and searching vectors.
   
   Regular DB:  SELECT * WHERE name = 'Python'     (exact match)
   Vector DB:   Find vectors CLOSEST to this vector (similarity)
   
   FAISS = Facebook AI Similarity Search
   - Open source, free
   - Very fast (millions of vectors)
   - Works locally (no cloud needed)
    """)
    
    # Create sample documents
    documents = [
        "Python developer with 5 years Django REST API experience",
        "Java developer skilled in Spring Boot and microservices",
        "Frontend developer expert in React TypeScript and CSS",
        "Network administrator managing Cisco and firewall security",
        "Database administrator with Oracle and PostgreSQL experience",
        "Python data scientist using machine learning and TensorFlow",
        "DevOps engineer skilled in Docker Kubernetes and AWS",
        "Security analyst performing penetration testing and audits",
        "Project manager with agile scrum methodology experience",
        "Full stack developer with Node.js React and MongoDB",
    ]
    
    metadatas = [
        {"job": "Python_Developer"}, {"job": "Java_Developer"},
        {"job": "Front_End_Developer"}, {"job": "Network_Administrator"},
        {"job": "Database_Administrator"}, {"job": "Python_Developer"},
        {"job": "Systems_Administrator"}, {"job": "Security_Analyst"},
        {"job": "Project_manager"}, {"job": "Web_Developer"},
    ]
    
    # Create FAISS index
    print("   Creating FAISS index...")
    
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    lc_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(documents, metadatas)
    ]
    
    store = FAISS.from_documents(documents=lc_docs, embedding=embeddings)
    
    print(f"   Documents stored: {len(documents)}")
    print(f"   Index type: FAISS (Facebook AI Similarity Search)")
    
    return store, embeddings, documents


# ============================================================
# PART 3: Semantic Search
# ============================================================

def demo_semantic_search(store):
    print_header("PART 3: SEMANTIC SEARCH")
    
    print("""
   Semantic Search = Find by MEANING, not keywords.
   
   Query: "I know Python and SQL"
   
   Keyword search: Looks for exact words "Python" and "SQL"
   Semantic search: Understands MEANING → finds related roles
    """)
    
    queries = [
        "I know Python and SQL, what job suits me?",
        "I want to build websites with JavaScript",
        "I have experience managing computer networks",
    ]
    
    for query in queries:
        print(f"\n   QUERY: \"{query}\"")
        print("   " + "-" * 50)
        
        results = store.similarity_search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results):
            sim = round(1 / (1 + score), 3) if score > 0 else 1.0
            bar = '█' * int(sim * 20) + '░' * (20 - int(sim * 20))
            print(f"   {i+1}. {bar} {sim:.3f}")
            print(f"      Job: {doc.metadata.get('job', 'N/A')}")
            print(f"      Text: {doc.page_content[:60]}...")
            print()


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("EMBEDDINGS + VECTOR STORE DEMO")
    
    model, embs = demo_embeddings()
    store, embeddings, docs = demo_vectorstore(model)
    demo_semantic_search(store)


if __name__ == "__main__":
    main()
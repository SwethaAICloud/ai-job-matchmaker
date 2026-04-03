# ============================================================
# 4_retrieval_methods.py
# 4 Retrieval Methods: Similarity, MMR, Metadata Filter, Hybrid
# ============================================================

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import EMBEDDING_MODEL


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


def setup_store():
    """Create FAISS store with sample data."""
    documents = [
        Document(page_content="Python developer Django REST APIs 5 years", metadata={"job": "Python_Developer", "years": "5", "level": "senior"}),
        Document(page_content="Python programmer Flask SQLAlchemy backend", metadata={"job": "Python_Developer", "years": "3", "level": "mid"}),
        Document(page_content="Python data scientist pandas numpy ML", metadata={"job": "Python_Developer", "years": "4", "level": "mid"}),
        Document(page_content="Java developer Spring Boot microservices", metadata={"job": "Java_Developer", "years": "6", "level": "senior"}),
        Document(page_content="React frontend developer TypeScript CSS", metadata={"job": "Front_End_Developer", "years": "2", "level": "junior"}),
        Document(page_content="Angular frontend developer JavaScript UI", metadata={"job": "Front_End_Developer", "years": "3", "level": "mid"}),
        Document(page_content="Network administrator Cisco firewall TCP", metadata={"job": "Network_Admin", "years": "7", "level": "senior"}),
        Document(page_content="Database admin Oracle PostgreSQL backup", metadata={"job": "Database_Admin", "years": "5", "level": "senior"}),
        Document(page_content="Security analyst penetration testing SIEM", metadata={"job": "Security_Analyst", "years": "4", "level": "mid"}),
        Document(page_content="DevOps Docker Kubernetes AWS CI/CD pipelines", metadata={"job": "Systems_Admin", "years": "3", "level": "mid"}),
        Document(page_content="Python developer machine learning TensorFlow", metadata={"job": "Python_Developer", "years": "2", "level": "junior"}),
        Document(page_content="Full stack Node.js React MongoDB developer", metadata={"job": "Web_Developer", "years": "4", "level": "mid"}),
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    store = FAISS.from_documents(documents=documents, embedding=embeddings)
    return store


def print_results(results, method_name):
    print(f"\n   Method: {method_name}")
    print("   " + "-" * 50)
    for i, doc in enumerate(results):
        job = doc.metadata.get('job', 'N/A')
        level = doc.metadata.get('level', 'N/A')
        print(f"   {i+1}. [{job}] [{level}] {doc.page_content[:50]}")
    print()


# ============================================================
# METHOD 1: Similarity Search
# ============================================================

def demo_similarity(store):
    print_header("METHOD 1: SIMILARITY SEARCH")
    print("""
   How it works:
   1. Convert query to vector
   2. Find K nearest vectors in FAISS
   3. Return closest matches
   
   Simple but may return DUPLICATE content
   (all results could be about same topic)
    """)
    
    query = "Python backend developer"
    results = store.similarity_search(query, k=5)
    print_results(results, "Similarity Search (k=5)")
    
    print("   NOTICE: Multiple Python results (similar content)")
    print("   This is the PROBLEM with pure similarity search.")


# ============================================================
# METHOD 2: Maximum Marginal Relevance (MMR)
# ============================================================

def demo_mmr(store):
    print_header("METHOD 2: MAXIMUM MARGINAL RELEVANCE (MMR)")
    print("""
   How it works:
   1. Find K nearest vectors (like similarity)
   2. BUT ALSO ensure DIVERSITY in results
   3. Penalizes documents too similar to already selected ones
   
   Formula:
   MMR = lambda * Sim(query, doc) - (1-lambda) * max(Sim(doc, selected))
   
   lambda = 0.5 → balance relevance and diversity
   lambda = 1.0 → pure similarity (no diversity)
   lambda = 0.0 → maximum diversity (less relevant)
   
   BEST METHOD for RAG (avoids redundant context)
    """)
    
    query = "Python backend developer"
    
    # MMR with diversity
    results = store.max_marginal_relevance_search(
        query, k=5, fetch_k=10, lambda_mult=0.5
    )
    print_results(results, "MMR (lambda=0.5, balanced)")
    
    # MMR with more diversity
    results2 = store.max_marginal_relevance_search(
        query, k=5, fetch_k=10, lambda_mult=0.2
    )
    print_results(results2, "MMR (lambda=0.2, more diverse)")
    
    print("   NOTICE: MMR returns DIVERSE results (not all Python)")
    print("   This gives the LLM BROADER context to work with.")


# ============================================================
# METHOD 3: Metadata Filtering
# ============================================================

def demo_metadata_filter(store):
    print_header("METHOD 3: METADATA FILTERING")
    print("""
   How it works:
   1. First FILTER by metadata (job, level, years, etc.)
   2. Then search within filtered results
   
   Like SQL WHERE clause + vector search:
   "Find similar to 'Python dev' WHERE level = 'senior'"
   
   GREAT for structured data like resumes.
    """)
    
    query = "Python developer"
    
    # Filter by job type
    print("   Filter: job = Python_Developer")
    results = store.similarity_search(
        query, k=3,
        filter={"job": "Python_Developer"}
    )
    print_results(results, "Filter by Python_Developer")
    
    # Filter by level
    print("   Filter: level = senior")
    results2 = store.similarity_search(
        query, k=3,
        filter={"level": "senior"}
    )
    print_results(results2, "Filter by Senior level")
    
    # Filter by level
    print("   Filter: level = junior")
    results3 = store.similarity_search(
        query, k=3,
        filter={"level": "junior"}
    )
    print_results(results3, "Filter by Junior level")


# ============================================================
# METHOD 4: Hybrid Search (Similarity + Keyword)
# ============================================================

def demo_hybrid():
    print_header("METHOD 4: HYBRID SEARCH (concept)")
    print("""
   How it works:
   Combines SEMANTIC search + KEYWORD search
   
   Semantic: "Python dev" finds "backend programmer" (meaning)
   Keyword:  "Python dev" finds "Python developer" (exact match)
   Hybrid:   Both combined → best results
   
   Score = alpha * semantic_score + (1-alpha) * keyword_score
   
   FAISS doesn't support hybrid natively.
   Use Elasticsearch or Pinecone for true hybrid.
   
   WORKAROUND with FAISS:
   1. Do semantic search → get top 20
   2. Re-rank by keyword match → return top 5
    """)
    
    print("   DEMO: Manual hybrid search\n")
    
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    docs = [
        "Python developer with Django REST API experience",
        "Backend programmer using Python and Flask",
        "Python data scientist machine learning expert",
        "Java Spring Boot microservices developer",
        "React frontend JavaScript TypeScript developer",
    ]
    
    query = "Python Django developer"
    
    # Semantic scores
    query_emb = embeddings.embed_query(query)
    doc_embs = embeddings.embed_documents(docs)
    
    from sklearn.metrics.pairwise import cosine_similarity
    semantic_scores = cosine_similarity([query_emb], doc_embs)[0]
    
    # Keyword scores
    query_words = set(query.lower().split())
    keyword_scores = []
    for doc in docs:
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        keyword_scores.append(overlap / len(query_words))
    
    # Hybrid scores
    alpha = 0.7
    hybrid_scores = [
        alpha * sem + (1 - alpha) * kw
        for sem, kw in zip(semantic_scores, keyword_scores)
    ]
    
    print(f"   Query: \"{query}\"")
    print(f"   Alpha: {alpha} (70% semantic, 30% keyword)\n")
    print(f"   {'Document':<50} {'Semantic':>9} {'Keyword':>9} {'Hybrid':>9}")
    print("   " + "-" * 80)
    
    ranked = sorted(range(len(docs)), key=lambda i: hybrid_scores[i], reverse=True)
    for i in ranked:
        print(f"   {docs[i][:48]:<50} {semantic_scores[i]:>8.3f} {keyword_scores[i]:>8.3f} {hybrid_scores[i]:>8.3f}")


# ============================================================
# COMPARISON
# ============================================================

def compare_methods():
    print_header("RETRIEVAL METHODS COMPARISON")
    
    print(f"\n   {'Method':<25} {'Speed':>8} {'Diversity':>10} {'Precision':>10} {'Best For':<25}")
    print("   " + "-" * 80)
    print(f"   {'Similarity':<25} {'Fast':>8} {'Low':>10} {'High':>10} {'Simple search':<25}")
    print(f"   {'MMR':<25} {'Medium':>8} {'High':>10} {'High':>10} {'RAG context (best)':<25}")
    print(f"   {'Metadata Filter':<25} {'Fast':>8} {'Medium':>10} {'High':>10} {'Structured data':<25}")
    print(f"   {'Hybrid':<25} {'Slow':>8} {'Medium':>10} {'Highest':>10} {'Production systems':<25}")
    
    print("\n   RECOMMENDATION FOR RAG:")
    print("   → Use MMR for getting diverse context")
    print("   → Add metadata filtering for structured queries")
    print("   → Hybrid if you need maximum precision")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading embedding model...")
    store = setup_store()
    
    demo_similarity(store)
    demo_mmr(store)
    demo_metadata_filter(store)
    demo_hybrid()
    compare_methods()


if __name__ == "__main__":
    main()
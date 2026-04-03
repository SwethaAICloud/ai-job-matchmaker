# ============================================================
# 1_chunking_methods.py
# Compare 6 different chunking strategies
# ============================================================

import re
import time

# Sample text (simulated resume)
SAMPLE_TEXT = """
John Smith is a Senior Python Developer with 8 years of experience.
He has expertise in Python, Django, Flask, REST APIs, and PostgreSQL.
He worked at Google for 3 years building scalable microservices.
Then he moved to Amazon where he led a team of 5 developers.
His education includes a Master's degree in Computer Science from MIT.
He is certified in AWS Solutions Architect and Docker.
His projects include building a real-time data pipeline processing
1 million events per day, and creating an open-source Python library
with 500+ GitHub stars. He is skilled in agile methodologies,
CI/CD pipelines, and test-driven development.
John is looking for a senior role in backend development
with focus on cloud architecture and distributed systems.
He prefers remote work and is open to relocation.
His salary expectation is between $150,000 and $180,000.
"""


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


def print_chunks(chunks, method_name):
    print(f"\n   Method: {method_name}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Avg chunk size: {sum(len(c) for c in chunks) // max(len(chunks),1)} chars")
    print("   " + "-" * 50)
    for i, chunk in enumerate(chunks):
        preview = chunk[:80].replace('\n', ' ').strip()
        print(f"   Chunk {i+1} ({len(chunk)} chars): {preview}...")
    print()


# ============================================================
# METHOD 1: Character Chunking (Fixed Size)
# ============================================================

def character_chunking(text, chunk_size=200, overlap=0):
    """
    Simplest method: Split every N characters.
    
    Problem: Cuts words and sentences in half.
    
    "Python Developer with 8 yea" | "rs of experience"
                              ^^^   ^^^
                         Word cut in half!
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


# ============================================================
# METHOD 2: Character Chunking with Overlap
# ============================================================

def character_overlap_chunking(text, chunk_size=200, overlap=50):
    """
    Same as character but with overlap.
    Overlap prevents losing context at boundaries.
    
    Chunk 1: "...Python Developer with 8 years of exper"
    Chunk 2: "years of experience in Django and Flask..."
                ^^^^^^^^^^^^^^^^
                Overlapping region (shared text)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c]


# ============================================================
# METHOD 3: Recursive Character Chunking
# ============================================================

def recursive_chunking(text, chunk_size=200, overlap=30):
    """
    LangChain's default method.
    Tries to split at natural boundaries:
      1. First try: Split at paragraphs (\\n\\n)
      2. Then: Split at sentences (\\n)
      3. Then: Split at periods (. )
      4. Then: Split at commas (, )
      5. Finally: Split at spaces ( )
    
    Best general-purpose method.
    """
    separators = ["\n\n", "\n", ". ", ", ", " "]
    chunks = []
    
    def split_text(text, seps):
        if len(text) <= chunk_size:
            return [text]
        
        for sep in seps:
            parts = text.split(sep)
            if len(parts) > 1:
                current = ""
                for part in parts:
                    if len(current) + len(part) + len(sep) <= chunk_size:
                        current += part + sep
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        current = part + sep
                if current.strip():
                    chunks.append(current.strip())
                return chunks
        
        # Fallback: character split
        return character_overlap_chunking(text, chunk_size, overlap)
    
    result = split_text(text, separators)
    return result if result else chunks


# ============================================================
# METHOD 4: Token-Based Chunking
# ============================================================

def token_chunking(text, max_tokens=50, overlap_tokens=10):
    """
    Split by word count (approximation of tokens).
    
    More natural than character splitting because
    it never cuts words in half.
    
    1 token ≈ 1 word (roughly)
    GPT counts tokens differently, but words are close enough.
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start = end - overlap_tokens
        if start >= len(words):
            break
    
    return [c for c in chunks if c]


# ============================================================
# METHOD 5: Sentence-Based Chunking
# ============================================================

def sentence_chunking(text, max_sentences=3, overlap_sentences=1):
    """
    Split at sentence boundaries.
    Never cuts a sentence in half.
    
    Best for maintaining meaning.
    
    Chunk 1: "Sentence 1. Sentence 2. Sentence 3."
    Chunk 2: "Sentence 3. Sentence 4. Sentence 5."
                          ^^^^^^^^^^
                        Overlap (shared sentence)
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    start = 0
    
    while start < len(sentences):
        end = start + max_sentences
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start = end - overlap_sentences
        if start >= len(sentences):
            break
    
    return [c for c in chunks if c]


# ============================================================
# METHOD 6: Semantic Chunking
# ============================================================

def semantic_chunking(text, similarity_threshold=0.5):
    """
    Split based on MEANING changes.
    
    Uses embeddings to detect when topic changes:
    - "Python Developer with 8 years..." (skills topic)
    - "He worked at Google for 3 years..." (experience topic)
    - "His education includes..." (education topic)
    
    When similarity between consecutive sentences drops
    below threshold, we start a new chunk.
    
    MOST INTELLIGENT method but slowest.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return [text]
    
    # Get embeddings for each sentence
    embeddings = model.encode(sentences)
    
    # Calculate similarity between consecutive sentences
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Cosine similarity between current and previous sentence
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
        )
        
        if sim < similarity_threshold:
            # Topic changed — start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            # Same topic — continue chunk
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# ============================================================
# METHOD 7: Structure-Based Chunking
# ============================================================

def structure_chunking(text):
    """
    Split based on document structure.
    Uses headers, sections, bullet points.
    
    Good for structured documents like resumes:
    - Skills section → one chunk
    - Experience section → one chunk
    - Education section → one chunk
    """
    # Define section markers
    section_markers = [
        'experience', 'education', 'skills', 'projects',
        'certifications', 'certified', 'salary', 'looking for'
    ]
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        # Check if this sentence starts a new section
        is_new_section = any(
            marker in sentence.lower()
            for marker in section_markers
        )
        
        if is_new_section and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# ============================================================
# COMPARE ALL METHODS
# ============================================================

def compare_all():
    print_header("CHUNKING METHODS COMPARISON")
    
    print("\n   INPUT TEXT:")
    print("   " + "-" * 50)
    print("   " + SAMPLE_TEXT[:200].replace('\n', ' ').strip() + "...")
    print(f"   Total length: {len(SAMPLE_TEXT)} characters")
    print(f"   Total words: {len(SAMPLE_TEXT.split())}")
    
    methods = [
        ("1. CHARACTER (fixed 200)", lambda: character_chunking(SAMPLE_TEXT, 200, 0)),
        ("2. CHARACTER + OVERLAP (200, 50)", lambda: character_overlap_chunking(SAMPLE_TEXT, 200, 50)),
        ("3. RECURSIVE (200, 30)", lambda: recursive_chunking(SAMPLE_TEXT, 200, 30)),
        ("4. TOKEN-BASED (50 words)", lambda: token_chunking(SAMPLE_TEXT, 50, 10)),
        ("5. SENTENCE-BASED (3 sentences)", lambda: sentence_chunking(SAMPLE_TEXT, 3, 1)),
        ("6. STRUCTURE-BASED", lambda: structure_chunking(SAMPLE_TEXT)),
    ]
    
    results = {}
    
    for name, method in methods:
        start = time.time()
        chunks = method()
        elapsed = time.time() - start
        print_chunks(chunks, name)
        results[name] = {
            'chunks': len(chunks),
            'avg_size': sum(len(c) for c in chunks) // max(len(chunks), 1),
            'time_ms': round(elapsed * 1000, 2)
        }
    
    # Semantic chunking (separate because it loads model)
    print("\n   Loading embedding model for semantic chunking...")
    start = time.time()
    chunks = semantic_chunking(SAMPLE_TEXT, 0.5)
    elapsed = time.time() - start
    name = "7. SEMANTIC (similarity < 0.5)"
    print_chunks(chunks, name)
    results[name] = {
        'chunks': len(chunks),
        'avg_size': sum(len(c) for c in chunks) // max(len(chunks), 1),
        'time_ms': round(elapsed * 1000, 2)
    }
    
    # Summary table
    print_header("COMPARISON SUMMARY")
    print(f"\n   {'Method':<35} {'Chunks':>7} {'Avg Size':>10} {'Time':>10}")
    print("   " + "-" * 65)
    
    for name, data in results.items():
        print(f"   {name:<35} {data['chunks']:>7} {str(data['avg_size'])+' ch':>10} {str(data['time_ms'])+' ms':>10}")
    
    print("\n   RECOMMENDATIONS:")
    print("   " + "-" * 50)
    print("   General purpose  → Recursive Character (LangChain default)")
    print("   Preserve meaning → Sentence-Based")
    print("   Best quality     → Semantic (but slowest)")
    print("   Structured docs  → Structure-Based")
    print("   Simple/fast      → Token-Based")
    print("   Avoid            → Character (cuts words)")


if __name__ == "__main__":
    compare_all()
# ============================================================
# 5_response_synthesis.py
# Combining Query + Context → LLM Response
# ============================================================

import time
from groq import Groq
from config import GROQ_API_KEY


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


def ask_llm(prompt, max_tokens=512):
    client = Groq(api_key=GROQ_API_KEY)
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.3
    )
    return r.choices[0].message.content.strip()


# ============================================================
# METHOD 1: Stuff (Put everything in one prompt)
# ============================================================

def demo_stuff():
    print_header("METHOD 1: STUFF (All context in one prompt)")
    print("""
   How: Put ALL retrieved documents into ONE prompt.
   
   Prompt = Question + Doc1 + Doc2 + Doc3 + Doc4 + Doc5
   
   Pros: Simple, preserves all context
   Cons: May exceed token limit with many docs
   Best: When context fits in one prompt (< 4000 tokens)
    """)
    
    question = "What role suits someone with Python and SQL skills?"
    
    contexts = [
        "Resume 1: Python developer with 5 years Django REST API experience, SQL expert",
        "Resume 2: Database administrator skilled in PostgreSQL and Python scripting",
        "Resume 3: Backend developer using Python Flask and MySQL databases",
    ]
    
    prompt = (
        "Based on the following resume data, answer the question.\n\n"
        "CONTEXT:\n" + "\n".join(contexts) + "\n\n"
        "QUESTION: " + question + "\n\n"
        "ANSWER:"
    )
    
    print(f"   Question: {question}")
    print(f"   Contexts: {len(contexts)} documents")
    print(f"   Prompt length: {len(prompt)} chars\n")
    print("   Generating answer...")
    
    answer = ask_llm(prompt)
    print(f"\n   ANSWER:\n   {answer[:300]}")
    time.sleep(3)


# ============================================================
# METHOD 2: Map-Reduce
# ============================================================

def demo_map_reduce():
    print_header("METHOD 2: MAP-REDUCE")
    print("""
   How:
   Step 1 (MAP):    Ask LLM about EACH document separately
   Step 2 (REDUCE): Combine all answers into final answer
   
   Doc1 → LLM → Answer1
   Doc2 → LLM → Answer2  → LLM → FINAL ANSWER
   Doc3 → LLM → Answer3
   
   Pros: Handles unlimited documents
   Cons: Slow (multiple LLM calls), expensive
   Best: When you have MANY documents
    """)
    
    question = "What role suits someone with Python and SQL?"
    
    contexts = [
        "Python developer with 5 years Django REST API experience",
        "Database administrator skilled in PostgreSQL and Python",
        "Backend developer using Python Flask and MySQL",
    ]
    
    # MAP step
    print("   STEP 1: MAP (process each document)\n")
    summaries = []
    for i, ctx in enumerate(contexts):
        print(f"   Processing doc {i+1}...")
        prompt = f"Based on this resume: '{ctx}', what job role is relevant for Python and SQL skills? Answer in one sentence."
        summary = ask_llm(prompt, max_tokens=100)
        summaries.append(summary)
        print(f"   Summary {i+1}: {summary[:100]}")
        time.sleep(3)
    
    # REDUCE step
    print(f"\n   STEP 2: REDUCE (combine summaries)\n")
    reduce_prompt = (
        "Combine these summaries into a final answer about what role suits Python and SQL:\n\n"
        + "\n".join(summaries) + "\n\n"
        "FINAL ANSWER:"
    )
    
    final = ask_llm(reduce_prompt)
    print(f"   FINAL: {final[:300]}")
    time.sleep(3)


# ============================================================
# METHOD 3: Refine
# ============================================================

def demo_refine():
    print_header("METHOD 3: REFINE")
    print("""
   How:
   Start with Doc1 → get initial answer
   Add Doc2 → REFINE the answer
   Add Doc3 → REFINE again
   
   Doc1 → LLM → Answer v1
   Doc2 + Answer v1 → LLM → Answer v2 (refined)
   Doc3 + Answer v2 → LLM → Answer v3 (final)
   
   Pros: Builds up answer progressively
   Cons: Slow, later docs may override earlier ones
   Best: When document ORDER matters
    """)
    
    contexts = [
        "Python developer with 5 years Django experience",
        "Database administrator skilled in PostgreSQL",
        "Backend developer using Flask and MySQL",
    ]
    
    question = "What role suits Python and SQL skills?"
    
    # Initial answer
    print("   Processing doc 1...")
    answer = ask_llm(
        f"Based on: '{contexts[0]}', answer: {question}",
        max_tokens=150
    )
    print(f"   v1: {answer[:100]}")
    time.sleep(3)
    
    # Refine with each subsequent doc
    for i, ctx in enumerate(contexts[1:], 2):
        print(f"\n   Refining with doc {i}...")
        refine_prompt = (
            f"Current answer: {answer}\n\n"
            f"New information: {ctx}\n\n"
            f"Refine the answer based on this new information. Question: {question}"
        )
        answer = ask_llm(refine_prompt, max_tokens=200)
        print(f"   v{i}: {answer[:100]}")
        time.sleep(3)
    
    print(f"\n   FINAL: {answer[:300]}")


# ============================================================
# COMPARISON
# ============================================================

def compare():
    print_header("RESPONSE SYNTHESIS COMPARISON")
    
    print(f"\n   {'Method':<15} {'LLM Calls':>10} {'Speed':>8} {'Quality':>10} {'Best For':<25}")
    print("   " + "-" * 70)
    print(f"   {'Stuff':<15} {'1':>10} {'Fast':>8} {'Good':>10} {'Small context (<4K)':<25}")
    print(f"   {'Map-Reduce':<15} {'N+1':>10} {'Slow':>8} {'Good':>10} {'Many documents':<25}")
    print(f"   {'Refine':<15} {'N':>10} {'Slow':>8} {'Best':>10} {'Ordered documents':<25}")
    
    print("\n   RECOMMENDATION:")
    print("   → Use STUFF for most RAG applications (simple, fast)")
    print("   → Use MAP-REDUCE only if context exceeds token limit")
    print("   → Use REFINE for sequential/ordered documents")


def main():
    demo_stuff()
    demo_map_reduce()
    demo_refine()
    compare()


if __name__ == "__main__":
    main()
# ============================================================
# 6_evaluation.py
# RAG Evaluation: Triad of Metrics
# Context Relevance + Groundedness + Answer Relevance
# ============================================================

import time
import json
import random
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODELS
import re


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


class Evaluator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODELS[0]

    def score(self, prompt):
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50, temperature=0.1
            )
            text = r.choices[0].message.content.strip()
            nums = re.findall(r'(\d+(?:\.\d+)?)', text)
            for n in nums:
                s = float(n)
                if 1 <= s <= 5:
                    return s
            return 3.0
        except:
            time.sleep(10)
            return 3.0


# ============================================================
# TRIAD OF METRICS — Explained
# ============================================================

def explain_triad():
    print_header("TRIAD OF METRICS — THE 3 KEY RAG METRICS")
    print("""
   The RAG Triad evaluates 3 things:

   ┌─────────────────────────────────────────────┐
   │                                             │
   │         1. CONTEXT RELEVANCE                │
   │    "Is the retrieved context relevant       │
   │     to the question?"                       │
   │                                             │
   │    Question ──→ Retriever ──→ Context        │
   │                     ↑                       │
   │              Is this good?                  │
   │                                             │
   ├─────────────────────────────────────────────┤
   │                                             │
   │         2. GROUNDEDNESS                     │
   │    "Is the answer based on the context?"    │
   │                                             │
   │    Context ──→ LLM ──→ Answer               │
   │                  ↑                          │
   │           Is this faithful?                 │
   │                                             │
   ├─────────────────────────────────────────────┤
   │                                             │
   │         3. ANSWER RELEVANCE                 │
   │    "Does the answer address the question?"  │
   │                                             │
   │    Question ──→ ... ──→ Answer               │
   │         ↑                  ↑                │
   │         └── Do these match? ┘               │
   │                                             │
   └─────────────────────────────────────────────┘

   ALL THREE must be high for a good RAG system:
   
   High Context Relevance + Low Groundedness = LLM ignores context
   High Groundedness + Low Context Relevance = Faithful to bad data
   High both + Low Answer Relevance = Right data, wrong answer
    """)


# ============================================================
# SAMPLE DATA FOR EVALUATION
# ============================================================

def create_eval_data():
    """Create sample RAG outputs to evaluate."""
    return [
        {
            "question": "What role suits someone with Python and SQL?",
            "context": "Resume: Python developer with 5 years Django REST API experience. Skills include Python, SQL, PostgreSQL, and AWS.",
            "answer": "Based on your Python and SQL skills, Python Developer is the best match (85%). Software Developer is also a strong option (72%). Focus on building REST API projects to strengthen your profile."
        },
        {
            "question": "I have 3 years React experience, what job fits?",
            "context": "Resume: Frontend developer expert in React TypeScript and CSS with 3 years building responsive web applications.",
            "answer": "With 3 years of React experience, Front End Developer is your best match. You should also consider Web Developer roles. Adding TypeScript and testing skills will boost your profile."
        },
        {
            "question": "Compare frontend vs backend development",
            "context": "Resume 1: React frontend developer with JavaScript. Resume 2: Python backend developer with Django. Frontend focuses on UI/UX while backend handles server logic and databases.",
            "answer": "Frontend development focuses on user interfaces using React/JavaScript, while backend focuses on server logic using Python/Django. Backend typically has higher salaries but frontend has more visual creativity."
        },
        {
            "question": "What salary can I expect with Java and 5 years?",
            "context": "Resume: Senior Java developer with Spring Boot, 6 years experience. Java developers with 5+ years typically work as senior developers at major tech companies.",
            "answer": "With Java and 5 years experience, expect $110,000-$140,000. Senior roles at FAANG companies can reach $180,000+. Certifications like Oracle Java can boost salary by 10-15%."
        },
        {
            "question": "I am a fresher with basic Python, what should I do?",
            "context": "Resume: Junior Python developer, 1 year internship, basic skills in Python and SQL. Entry level positions typically require portfolio projects.",
            "answer": "As a fresher, focus on: 1) Build 3-5 portfolio projects 2) Learn Django or Flask 3) Get AWS certification 4) Apply for junior/entry level positions. Your hire probability increases from 30% to 70% with projects."
        },
    ]


# ============================================================
# METRIC 1: CONTEXT RELEVANCE
# ============================================================

def eval_context_relevance(evaluator, data):
    print_header("METRIC 1: CONTEXT RELEVANCE")
    print("   Is the retrieved context relevant to the question?\n")
    
    scores = []
    for i, d in enumerate(data):
        print(f"   [{i+1}/{len(data)}] {d['question'][:50]}", end="")
        
        prompt = (
            "Rate how RELEVANT the context is to the question.\n\n"
            "QUESTION: " + d['question'] + "\n"
            "CONTEXT: " + d['context'][:300] + "\n\n"
            "Score 1-5 (5=perfectly relevant). Reply ONLY a number."
        )
        s = evaluator.score(prompt)
        scores.append(s)
        bar = '█' * int(s * 4) + '░' * (20 - int(s * 4))
        print(f"  {bar} {s}/5")
        time.sleep(3)
    
    avg = sum(scores) / len(scores)
    print(f"\n   AVERAGE CONTEXT RELEVANCE: {avg:.2f}/5 ({avg*20:.1f}%)")
    return scores, avg


# ============================================================
# METRIC 2: GROUNDEDNESS
# ============================================================

def eval_groundedness(evaluator, data):
    print_header("METRIC 2: GROUNDEDNESS")
    print("   Is the answer grounded in (supported by) the context?\n")
    
    scores = []
    for i, d in enumerate(data):
        print(f"   [{i+1}/{len(data)}] {d['question'][:50]}", end="")
        
        prompt = (
            "Rate how GROUNDED the answer is in the context.\n"
            "5=fully supported, 1=completely hallucinated.\n\n"
            "CONTEXT: " + d['context'][:300] + "\n"
            "ANSWER: " + d['answer'][:300] + "\n\n"
            "Reply ONLY a number 1-5."
        )
        s = evaluator.score(prompt)
        scores.append(s)
        bar = '█' * int(s * 4) + '░' * (20 - int(s * 4))
        print(f"  {bar} {s}/5")
        time.sleep(3)
    
    avg = sum(scores) / len(scores)
    print(f"\n   AVERAGE GROUNDEDNESS: {avg:.2f}/5 ({avg*20:.1f}%)")
    return scores, avg


# ============================================================
# METRIC 3: ANSWER RELEVANCE
# ============================================================

def eval_answer_relevance(evaluator, data):
    print_header("METRIC 3: ANSWER RELEVANCE")
    print("   Does the answer address the question?\n")
    
    scores = []
    for i, d in enumerate(data):
        print(f"   [{i+1}/{len(data)}] {d['question'][:50]}", end="")
        
        prompt = (
            "Rate how well the answer addresses the question.\n"
            "5=perfectly answers, 1=completely irrelevant.\n\n"
            "QUESTION: " + d['question'] + "\n"
            "ANSWER: " + d['answer'][:300] + "\n\n"
            "Reply ONLY a number 1-5."
        )
        s = evaluator.score(prompt)
        scores.append(s)
        bar = '█' * int(s * 4) + '░' * (20 - int(s * 4))
        print(f"  {bar} {s}/5")
        time.sleep(3)
    
    avg = sum(scores) / len(scores)
    print(f"\n   AVERAGE ANSWER RELEVANCE: {avg:.2f}/5 ({avg*20:.1f}%)")
    return scores, avg


# ============================================================
# FINAL REPORT
# ============================================================

def final_report(ctx_avg, grd_avg, ans_avg, data):
    print_header("FINAL EVALUATION REPORT")
    
    overall = (ctx_avg + grd_avg + ans_avg) / 3
    
    metrics = [
        ("CONTEXT RELEVANCE", ctx_avg, "Retrieved docs relevant?"),
        ("GROUNDEDNESS", grd_avg, "Answer based on context?"),
        ("ANSWER RELEVANCE", ans_avg, "Answer addresses question?"),
    ]
    
    for name, score, desc in metrics:
        pct = score * 20
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"\n   {name}")
        print(f"   {desc}")
        print(f"   {bar}  {score:.2f}/5.0 ({pct:.1f}%)")
    
    overall_pct = overall * 20
    bar = '█' * int(overall_pct / 5) + '░' * (20 - int(overall_pct / 5))
    print(f"\n   {'─' * 50}")
    print(f"\n   OVERALL RAG QUALITY")
    print(f"   {bar}  {overall:.2f}/5.0 ({overall_pct:.1f}%)")
    
    print(f"\n   GRADE: ", end="")
    if overall_pct >= 90: print("A+ (Excellent)")
    elif overall_pct >= 80: print("A (Very Good)")
    elif overall_pct >= 70: print("B (Good)")
    elif overall_pct >= 60: print("C (Satisfactory)")
    else: print("D (Needs Improvement)")
    
    # Save
    results = {
        "questions_evaluated": len(data),
        "context_relevance": round(ctx_avg, 3),
        "groundedness": round(grd_avg, 3),
        "answer_relevance": round(ans_avg, 3),
        "overall": round(overall, 3),
        "overall_pct": round(overall_pct, 1),
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n   Saved to: evaluation_results.json")
    return overall


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("RAG EVALUATION — TRIAD OF METRICS")
    
    explain_triad()
    
    print("\n   Starting evaluation...\n")
    data = create_eval_data
    # ============================================================
# MAIN
# ============================================================

def main():
    print_header("RAG EVALUATION — TRIAD OF METRICS")
    
    explain_triad()
    
    input("\n   Press Enter to start evaluation...")
    
    data = create_eval_data()
    evaluator = Evaluator()
    
    ctx_scores, ctx_avg = eval_context_relevance(evaluator, data)
    grd_scores, grd_avg = eval_groundedness(evaluator, data)
    ans_scores, ans_avg = eval_answer_relevance(evaluator, data)
    
    overall = final_report(ctx_avg, grd_avg, ans_avg, data)
    
    # Per-question breakdown
    print_header("PER-QUESTION BREAKDOWN")
    print(f"\n   {'Question':<40} {'Context':>8} {'Ground':>8} {'Answer':>8} {'Avg':>8}")
    print("   " + "-" * 75)
    
    for i, d in enumerate(data):
        q = d['question'][:38]
        avg = (ctx_scores[i] + grd_scores[i] + ans_scores[i]) / 3
        print(f"   {q:<40} {ctx_scores[i]:>6.1f}/5 {grd_scores[i]:>6.1f}/5 {ans_scores[i]:>6.1f}/5 {avg:>6.1f}/5")


if __name__ == "__main__":
    main()
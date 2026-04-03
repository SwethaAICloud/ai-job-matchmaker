import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from datasets import Dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from config import GROQ_API_KEY, EMBEDDING_MODEL, GROQ_MODELS


def print_header(title):
    print("\n" + "=" * 55)
    print("   " + title)
    print("=" * 55)


# ============================================================
# STEP 1: Create Test Data
# ============================================================

def create_test_data():
    """Create questions with known answers for RAGAS."""
    return [
        {
            "question": "Which role suits me if I know Python and SQL?",
            "ground_truth": "Python Developer or Software Developer or Database Administrator are suitable roles for someone with Python and SQL skills."
        },
        {
            "question": "I know JavaScript, React, HTML, CSS - what job?",
            "ground_truth": "Front End Developer or Web Developer roles are best for JavaScript, React, HTML, and CSS skills."
        },
        {
            "question": "Java, Spring Boot, Microservices - suggest role",
            "ground_truth": "Java Developer or Software Developer roles are ideal for Java, Spring Boot, and Microservices experience."
        },
        {
            "question": "I have experience in machine learning and Python",
            "ground_truth": "Python Developer or ML Engineer roles suit someone with machine learning and Python experience."
        },
        {
            "question": "Network security and firewall management skills",
            "ground_truth": "Security Analyst or Network Administrator roles match network security and firewall skills."
        },
        {
            "question": "Django and REST APIs, what job should I apply for?",
            "ground_truth": "Python Developer or Web Developer roles are suitable for Django and REST API skills."
        },
        {
            "question": "I manage databases, Oracle and PostgreSQL expert",
            "ground_truth": "Database Administrator is the ideal role for Oracle and PostgreSQL expertise."
        },
        {
            "question": "Linux, Docker, Kubernetes, CI/CD - what role?",
            "ground_truth": "Systems Administrator or DevOps Engineer roles match Linux, Docker, Kubernetes, and CI/CD skills."
        },
        {
            "question": "Project management with Agile and Scrum experience",
            "ground_truth": "Project Manager role is the best fit for Agile and Scrum project management experience."
        },
        {
            "question": "Node.js, Express, MongoDB, full stack development",
            "ground_truth": "Web Developer or Full Stack Developer roles suit Node.js, Express, and MongoDB skills."
        },
    ]


# ============================================================
# STEP 2: Run Chatbot and Collect Results
# ============================================================

def run_chatbot_on_questions(test_data):
    """Run each question through chatbot, collect answers and contexts."""

    print("\nLoading chatbot components...")

    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    store = FAISS.load_local(
        "faiss_index", emb,
        allow_dangerous_deserialization=True
    )

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )

    client = Groq(api_key=GROQ_API_KEY)
    model = GROQ_MODELS[0]

    print("Ready!\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]

        print("   [" + str(i+1) + "/" + str(len(test_data)) + "] " + q[:50])

        # Retrieve context
        docs = retriever.invoke(q)
        ctx = [doc.page_content for doc in docs]

        # Generate answer
        context_text = "\n\n".join(ctx)
        prompt = (
            "You are a Career Advisor. Based on the context, answer the question.\n\n"
            "CONTEXT:\n" + context_text + "\n\n"
            "QUESTION: " + q + "\n\n"
            "Recommend matching IT job roles. Be specific.\n\nANSWER:"
        )

        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            answer = r.choices[0].message.content.strip()
        except:
            time.sleep(10)
            answer = "Error generating answer"

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

        time.sleep(3)

    return questions, answers, contexts, ground_truths


# ============================================================
# STEP 3: RAGAS Evaluation
# ============================================================

def run_ragas_evaluation(questions, answers, contexts, ground_truths):
    """Run official RAGAS evaluation."""

    print_header("RAGAS EVALUATION")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        # Create RAGAS dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(data)

        print("\nRunning RAGAS metrics...")
        print("This takes 5-10 minutes...\n")

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        return result

    except ImportError:
        print("RAGAS not installed. Install with: pip install ragas")
        return None
    except Exception as e:
        print("RAGAS error: " + str(e)[:200])
        print("\nFalling back to manual evaluation...")
        return None


# ============================================================
# STEP 4: Manual RAGAS-Style Evaluation (Fallback)
# ============================================================

def manual_ragas_evaluation(questions, answers, contexts, ground_truths):
    """Manual evaluation using same RAGAS metrics with Groq as judge."""

    print_header("MANUAL RAGAS-STYLE EVALUATION")
    print("   Using Groq as evaluation judge\n")

    client = Groq(api_key=GROQ_API_KEY)
    model = GROQ_MODELS[0]

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(len(questions)):
        print("   Evaluating " + str(i+1) + "/" + str(len(questions)), end="\r")

        q = questions[i]
        a = answers[i]
        ctx = " ".join(contexts[i])[:500]
        gt = ground_truths[i]

        # Metric 1: Faithfulness
        prompt1 = (
            "Rate FAITHFULNESS: Is the answer based on the given context?\n"
            "Context: " + ctx[:300] + "\n"
            "Answer: " + a[:300] + "\n"
            "Score 0.0 to 1.0 (1.0 = fully faithful). Reply ONLY a number."
        )

        # Metric 2: Answer Relevancy
        prompt2 = (
            "Rate ANSWER RELEVANCY: Does the answer address the question?\n"
            "Question: " + q + "\n"
            "Answer: " + a[:300] + "\n"
            "Score 0.0 to 1.0 (1.0 = perfectly relevant). Reply ONLY a number."
        )

        # Metric 3: Context Precision
        prompt3 = (
            "Rate CONTEXT PRECISION: Is the retrieved context relevant to the question?\n"
            "Question: " + q + "\n"
            "Context: " + ctx[:300] + "\n"
            "Score 0.0 to 1.0 (1.0 = perfectly relevant). Reply ONLY a number."
        )

        # Metric 4: Context Recall
        prompt4 = (
            "Rate CONTEXT RECALL: Does the context contain info needed to answer correctly?\n"
            "Question: " + q + "\n"
            "Context: " + ctx[:300] + "\n"
            "Expected Answer: " + gt + "\n"
            "Score 0.0 to 1.0 (1.0 = all info present). Reply ONLY a number."
        )

        for prompt, scores_list in [
            (prompt1, faithfulness_scores),
            (prompt2, relevancy_scores),
            (prompt3, precision_scores),
            (prompt4, recall_scores)
        ]:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                text = r.choices[0].message.content.strip()
                import re
                nums = re.findall(r'(\d+\.?\d*)', text)
                score = float(nums[0]) if nums else 0.5
                score = min(max(score, 0.0), 1.0)
                scores_list.append(score)
            except:
                scores_list.append(0.5)
            time.sleep(2)

    print("\n")

    results = {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores),
        "context_precision": sum(precision_scores) / len(precision_scores),
        "context_recall": sum(recall_scores) / len(recall_scores),
    }

    return results


# ============================================================
# STEP 5: Print Results
# ============================================================

def print_results(results, method="RAGAS"):
    print_header(method + " EVALUATION RESULTS")

    if hasattr(results, '__iter__') and not isinstance(results, dict):
        results = dict(results)

    metrics = [
        ("Faithfulness", results.get("faithfulness", 0), "Is answer based on retrieved context?"),
        ("Answer Relevancy", results.get("answer_relevancy", 0), "Does answer address the question?"),
        ("Context Precision", results.get("context_precision", 0), "Are retrieved docs relevant?"),
        ("Context Recall", results.get("context_recall", 0), "Did we retrieve all needed info?"),
    ]

    total = 0
    count = 0

    for name, score, desc in metrics:
        if isinstance(score, (int, float)):
            pct = score * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            print("\n   " + name)
            print("   " + desc)
            print("   [" + bar + "]  " + str(round(pct, 1)) + "%")
            total += pct
            count += 1

    if count > 0:
        overall = total / count
        bar = "#" * int(overall / 5) + "." * (20 - int(overall / 5))
        print("\n   " + "-" * 50)
        print("\n   OVERALL RAGAS SCORE")
        print("   [" + bar + "]  " + str(round(overall, 1)) + "%")

        if overall >= 90:
            grade = "A+ (Excellent)"
        elif overall >= 80:
            grade = "A (Very Good)"
        elif overall >= 70:
            grade = "B (Good)"
        elif overall >= 60:
            grade = "C (Satisfactory)"
        else:
            grade = "D (Needs Improvement)"

        print("   GRADE: " + grade)

    # Save results
    import json
    output = {}
    for name, score, desc in metrics:
        if isinstance(score, (int, float)):
            output[name.lower().replace(" ", "_")] = round(score, 4)
    output["overall"] = round(overall, 4) if count > 0 else 0

    with open("ragas_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n   Saved to: ragas_results.json")


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("RAGAS EVALUATION FOR RAG CHATBOT")
    print("""
   RAGAS = Retrieval Augmented Generation Assessment
   Industry-standard evaluation framework.

   Metrics:
   1. Faithfulness      - Is answer based on context?
   2. Answer Relevancy  - Does answer address question?
   3. Context Precision - Are retrieved docs relevant?
   4. Context Recall    - Did we find all needed info?

   Options:
   1. Full RAGAS evaluation (official library) - 10 min
   2. Manual RAGAS-style evaluation (Groq judge) - 5 min
   3. Both
   4. Exit
    """)

    choice = input("   Choice (1-4): ").strip()

    if choice in ["1", "2", "3"]:
        test_data = create_test_data()
        print("\nRunning " + str(len(test_data)) + " test questions...\n")

        questions, answers, contexts, ground_truths = run_chatbot_on_questions(test_data)

        if choice == "1" or choice == "3":
            result = run_ragas_evaluation(questions, answers, contexts, ground_truths)
            if result:
                print_results(result, "OFFICIAL RAGAS")
            else:
                print("   Official RAGAS failed. Running manual instead...")
                result = manual_ragas_evaluation(questions, answers, contexts, ground_truths)
                print_results(result, "MANUAL RAGAS-STYLE")

        if choice == "2":
            result = manual_ragas_evaluation(questions, answers, contexts, ground_truths)
            print_results(result, "MANUAL RAGAS-STYLE")

        if choice == "3" and result:
            manual_result = manual_ragas_evaluation(questions, answers, contexts, ground_truths)
            print_results(manual_result, "MANUAL RAGAS-STYLE")

    elif choice == "4":
        print("   Bye!")


if __name__ == "__main__":
    main()